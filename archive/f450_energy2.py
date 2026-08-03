"""
F450EnergyModel
---------------
Physically grounded energy and SOC calculation for a family of F450-class
drones with variable battery capacity (8000 / 12000 / 16000 / 20000 mAh,
3S LiPo) and variable payload that is dropped progressively along the route.

Physical basis (Abeywickrama et al., 2018 -- simplified form),
extended for:
  1. Dynamic mass  : payload mass decreases after each task drop, so
                      travel / climb / hover power for every leg use the
                      CURRENT (remaining) all-up-weight (AUW), not a
                      fixed constant.
  2. Variable battery: larger drones carry bigger packs
                      (8000/12000/16000/20000 mAh @ 11.1V -> 88.8/133.2/
                      177.6/222.0 Wh).
  3. Mass-scaled hover power: hover thrust must equal weight, and
                      hover power for a rotorcraft scales roughly with
                      weight^1.5 (P ~ T^1.5 from momentum theory, T = W).
                      So a heavier (loaded / bigger-frame) drone hovers
                      at higher power, not the fixed 266 W measured only
                      for the base 1.5 kg AUW.

Reference point (base F450, AUW = 1.5 kg, 8000 mAh pack):
  - Battery                 : 8000 mAh x 11.1 V (3S LiPo) = 88.8 Wh
  - Measured hover endurance: ~20 min
    -> P_hover(base AUW)     = 88.8 Wh / (20/60 h) = 266 W

Mass-scaled hover power:
  P_hover(m) = P_HOVER_BASE_W * (m / MASS_BASE_KG) ** 1.5

Energy per leg (mass m_leg = current AUW during that leg):
  Travel:  P_v   = P_hover(m_leg) + k(m_leg) * v_ms**3
           E     = P_v * (dist_m / v_ms) / 3600                    [Wh]
  Climb:   E     = m_leg * g * dh_m / (MOTOR_EFF * 3600)            [Wh]
  Service: E     = P_hover(m_leg) * t_service_s / 3600              [Wh]

SOC drop [%] = E_total_Wh / BATTERY_Wh(capacity) * 100

All operations are batched and vectorised (no Python loops) except for
an unavoidable cumulative sum over the sequence dimension to build the
"remaining mass per leg" profile (implemented with torch.cumsum, still
loop-free).
"""

import torch


class F450EnergyModel:
    """
    Stateless helper: call compute_batch() on each forward pass.
    Not an nn.Module -- just physical constants + tensor math.
    """

    # ---- Base (reference) physical constants -----------------------------
    MASS_BASE_KG      = 1.5          # base AUW used to measure P_HOVER_BASE_W [kg]
    G_MS2             = 9.81         # gravitational acceleration [m/s^2]
    MOTOR_EFF         = 0.85         # motor + propeller efficiency (climb only)
    P_HOVER_BASE_W    = 266.0        # measured battery draw at hover, AT MASS_BASE_KG [W]
    HOVER_POWER_EXP   = 1.5          # P_hover ~ mass^1.5 (momentum-theory thrust scaling)

    # ---- Variable battery packs (3S LiPo, 11.1 V) -------------------------
    CELL_VOLTAGE_V    = 11.1
    VALID_CAPACITIES_mAh = (8000, 12000, 16000, 20000)

    # ---- Coordinate space scaling ------------------------------------------
    SPACE_SCALE   = 100.0     # m per normalised distance unit
    SPEED_SCALE   = 10.0      # (m/s) per normalised speed unit

    # ------------------------------------------------------------------
    @classmethod
    def battery_Wh_from_capacity(cls, capacity_mAh):
        """
        Convert battery capacity (mAh) to energy (Wh).
        capacity_mAh : scalar, list, np.array, or torch tensor, values in
                       VALID_CAPACITIES_mAh (8000/12000/16000/20000).
        Returns a torch tensor of the same shape, in Wh.
        """
        if not torch.is_tensor(capacity_mAh):
            capacity_mAh = torch.as_tensor(capacity_mAh, dtype=torch.float32)
        return capacity_mAh.float() * cls.CELL_VOLTAGE_V / 1000.0

    @classmethod
    def hover_power_w(cls, mass_kg, battery_capacity_mAh=None):
        """
        Mass-scaled hover power [W].

        IMPORTANT calibration fix: the ^1.5 momentum-theory scaling is only
        valid relative to a drone's OWN frame/rotor class. A bigger battery
        (12000/16000/20000 mAh) implies a bigger frame with bigger rotors
        built to carry more weight -- it should NOT be judged against the
        tiny 1.5 kg / 266 W F450 baseline, or any real, legally-loaded
        heavy-lift drone reads back an absurd hover power (this was the
        source of >100% SOC drop even on capacity-compliant schedules).

        Instead, each battery tier gets its OWN baseline mass and hover
        power, scaled linearly with pack size (bigger pack -> proportionally
        bigger frame -> proportionally higher baseline hover power, same
        ~20 min baseline endurance across tiers). The ^1.5 exponent then
        only penalizes being loaded BEYOND that tier's own baseline, which
        is the physically meaningful comparison.

        mass_kg               : tensor, current AUW (any shape)
        battery_capacity_mAh  : tensor broadcastable to mass_kg's shape,
                                 or None to fall back to the single F450
                                 baseline (backward compatible).
        """
        if battery_capacity_mAh is None:
            mass_base = cls.MASS_BASE_KG
            power_base = cls.P_HOVER_BASE_W
        else:
            tier_scale = battery_capacity_mAh / cls.VALID_CAPACITIES_mAh[0]  # capacity/8000
            mass_base = cls.MASS_BASE_KG * tier_scale
            power_base = cls.P_HOVER_BASE_W * tier_scale

        return power_base * (mass_kg / mass_base) ** cls.HOVER_POWER_EXP

    # ------------------------------------------------------------------
    @classmethod
    def compute_batch(
        cls,
        completed_coords,     # [B, seq_len+1, 3]  normalised
        speeds,                # [B, seq_len]       normalised
        service_times,         # [B, seq_len]       seconds
        payload_drop_kg,       # [B, seq_len]       kg dropped AT each waypoint
                                #                    (0 for the return-to-depot leg)
        robot_own_weight_kg,   # [B]                 frame + battery + electronics, no payload
        initial_payload_kg,    # [B]                 total payload carried at mission start
        battery_capacity_mAh=None,  # [B]             pack size, used to pick the hover-power
                                     #                 baseline tier (see hover_power_w). None
                                     #                 falls back to the single F450 baseline.
    ):
        """
        Vectorised, loop-free energy calculation over a full batch, with
        dynamically decreasing AUW as payload is dropped along the route.

        Parameters
        ----------
        completed_coords : [B, seq_len+1, 3]
            Tour coordinates with return-to-depot appended as the last row.
            First row = depot; rows 1..seq_len = tasks in visit order.
        speeds : [B, seq_len]
            Chosen speed per leg in normalised units [0.5, 1.5].
        service_times : [B, seq_len]
            Service duration per waypoint in seconds; depot entry = 0.
        payload_drop_kg : [B, seq_len]
            Payload mass dropped upon arrival/service at waypoint i
            (index i corresponds to the leg ARRIVING at that waypoint,
            i.e. same indexing as `speeds`/`service_times`). The drop is
            assumed to happen after that leg's travel+service, so it only
            reduces mass for legs AFTER it (cumulative). Use 0 for the
            final return-to-depot leg (no task there).
        robot_own_weight_kg : [B]
            Fixed frame/battery/electronics weight (no payload), i.e. the
            AUW floor once every task's payload is dropped.
        initial_payload_kg : [B]
            Total payload mass carried at departure (should equal
            payload_drop_kg.sum(dim=1) for a self-consistent scenario;
            not enforced here so partial/infeasible routes still compute).

        Returns
        -------
        total_Wh : [B]  total energy consumed in Wh
        """
        # ---- Physical-unit conversions ----------------------------------
        seg_vec  = completed_coords[:, 1:, :] - completed_coords[:, :-1, :]
        dist_m   = torch.norm(seg_vec, dim=2) * cls.SPACE_SCALE
        climb_m  = torch.clamp(seg_vec[:, :, 2], min=0.0) * cls.SPACE_SCALE
        v_ms     = speeds * cls.SPEED_SCALE

        # ---- Dynamic AUW per leg ------------------------------------------
        # Mass BEFORE leg i = own_weight + initial_payload - (payload already
        # dropped at waypoints 0..i-1). Waypoint i's own drop happens AFTER
        # arriving/servicing at i, so it affects leg i+1 onward, not leg i.
        cum_dropped_before_leg = torch.cumsum(payload_drop_kg, dim=1) - payload_drop_kg
        remaining_payload = torch.clamp(
            initial_payload_kg.unsqueeze(1) - cum_dropped_before_leg, min=0.0
        )
        mass_leg = robot_own_weight_kg.unsqueeze(1) + remaining_payload  # [B, seq_len]

        # ---- Travel energy  E = P_hover(m) * t_flight + drag term  [Wh] ---
        t_flight_s = dist_m / v_ms
        cap_per_leg = battery_capacity_mAh.unsqueeze(1) if battery_capacity_mAh is not None else None
        P_hover_leg = cls.hover_power_w(mass_leg, cap_per_leg)         # [B, seq]
        # k tuned so that at v_max (15 m/s), power is ~2x hover power,
        # scaled per-leg to the mass-dependent hover power at that leg.
        k = P_hover_leg / (15.0 ** 3)                                  # [B, seq]
        P_v = P_hover_leg + k * (v_ms ** 3)                            # [B, seq]
        E_travel_Wh = (P_v / 3600.0) * t_flight_s

        # ---- Climb energy  E = m g dh / (eta * 3600)  [Wh] -----------------
        E_climb_Wh = (mass_leg * cls.G_MS2 * climb_m) / (cls.MOTOR_EFF * 3600.0)

        # ---- Service / hover energy  E = P_hover(m) * t_service  [Wh] ------
        E_service_Wh = (P_hover_leg / 3600.0) * service_times

        # ---- Sum over all legs ---------------------------------------------
        total_Wh = (E_travel_Wh + E_climb_Wh + E_service_Wh).sum(dim=1)

        return total_Wh

    @classmethod
    def compute_batch_with_soc(
        cls,
        completed_coords,
        speeds,
        service_times,
        payload_drop_kg,
        robot_own_weight_kg,
        initial_payload_kg,
        battery_capacity_mAh,
    ):
        """
        Convenience wrapper: compute_batch() + SOC drop against the
        drone's OWN battery pack size (8000/12000/16000/20000 mAh).

        battery_capacity_mAh : [B] tensor/array of per-robot pack sizes.
        """
        total_Wh = cls.compute_batch(
            completed_coords, speeds, service_times,
            payload_drop_kg, robot_own_weight_kg, initial_payload_kg,
            battery_capacity_mAh,
        )
        battery_Wh = cls.battery_Wh_from_capacity(battery_capacity_mAh)
        soc_drop = (total_Wh / battery_Wh) * 100.0
        return total_Wh, soc_drop

    @classmethod
    def battery_exceeded(cls, soc_drop):
        """Returns True where the drone would run out of battery mid-mission."""
        return soc_drop > 100.0