"""
GeneralUAVEnergyModel
----------------------
A generalized, physically-grounded energy consumption model for
heterogeneous multi-rotor UAVs, returning ENERGY (Wh) directly rather
than SOC% drop. No drone-category-specific constants (e.g. F450 266W
baseline) are hardcoded -- all physical parameters are passed in per
robot/batch, so the same model works across arbitrary UAV categories.

Core physics (per leg):
  Hover power   : P_hover(m) = k_thrust * (m * g) ** 1.5      [W]
                  (momentum theory: T = W, P ~ T^1.5, with a
                   robot-specific k_thrust absorbing rotor efficiency,
                   disk area, air density, etc.)
  Travel power  : P_travel(m, v) = P_hover(m) + k_drag * v**3  [W]
  Climb power   : P_climb(m) = m * g * v_climb / eta            [W]
                  (only over the climbing portion of a leg)
  Service power : P_hover(m) held during on-station service time

Energy per leg = integral of power over time for that leg's phases,
summed as Wh.

Mass per leg is dynamic: it decreases after each payload drop,
computed via cumulative sum over the route (vectorised, no Python loop
over waypoints).
"""

import torch


class GeneralUAVEnergyModel:
    """
    Stateless helper -- call compute_batch() on each forward pass.
    All physical parameters are passed in (per-robot tensors), so no
    single drone category's numbers are baked into the class.
    """

    G_MS2 = 9.81  # gravitational acceleration [m/s^2], universal constant

    # ------------------------------------------------------------------
    @classmethod
    def hover_power_w(cls, mass_kg, k_thrust):
        """
        Generic mass-scaled hover power [W].

        mass_kg   : [..., ] current AUW (any broadcastable shape)
        k_thrust  : [..., ] robot-specific thrust/power coefficient,
                    broadcastable to mass_kg. Absorbs rotor efficiency,
                    disk loading, air density, motor/ESC losses, etc.
                    Larger, more efficient airframes get a smaller
                    k_thrust; small/inefficient ones get a larger one.
                    This must be supplied per robot (e.g. derived from
                    a manufacturer hover-endurance spec), NOT assumed.

        P_hover = k_thrust * (mass_kg * G)^1.5
        """
        return k_thrust * (mass_kg * cls.G_MS2) ** 1.5

    # ------------------------------------------------------------------
    @classmethod
    def compute_batch(
        cls,
        completed_coords,     # [B, seq_len+1, 3]  normalised route coords
        speeds,                # [B, seq_len]       normalised speed per leg
        service_times,         # [B, seq_len]       seconds held at each waypoint
        payload_drop_kg,       # [B, seq_len]       kg dropped AT each waypoint
        robot_own_weight_kg,   # [B]                 frame+battery+electronics
        initial_payload_kg,    # [B]                 payload at mission start
        k_thrust,               # [B]                 per-robot hover coefficient
        k_drag,                 # [B]                 per-robot drag coefficient
        motor_eff,              # [B]                 climb motor+prop efficiency (0-1)
        space_scale=100.0,      # m per normalised distance unit
        speed_scale=10.0,       # m/s per normalised speed unit
    ):
        """
        Vectorised, loop-free energy calculation with dynamically
        decreasing AUW as payload is dropped along the route.

        Returns
        -------
        total_Wh   : [B]           total energy consumed
        leg_Wh     : [B, seq_len]  per-leg energy breakdown (travel+climb+service)
        """
        # ---- Physical-unit conversions ----------------------------------
        seg_vec = completed_coords[:, 1:, :] - completed_coords[:, :-1, :]
        dist_m = torch.norm(seg_vec, dim=2) * space_scale
        climb_m = torch.clamp(seg_vec[:, :, 2], min=0.0) * space_scale
        v_ms = speeds * speed_scale

        # ---- Dynamic AUW per leg ------------------------------------------
        cum_dropped_before_leg = torch.cumsum(payload_drop_kg, dim=1) - payload_drop_kg
        remaining_payload = torch.clamp(
            initial_payload_kg.unsqueeze(1) - cum_dropped_before_leg, min=0.0
        )
        mass_leg = robot_own_weight_kg.unsqueeze(1) + remaining_payload  # [B, seq_len]

        # ---- Per-robot coefficients broadcast to per-leg ---------------
        k_thrust_leg = k_thrust.unsqueeze(1)   # [B, 1] -> broadcasts over seq
        k_drag_leg = k_drag.unsqueeze(1)
        motor_eff_leg = motor_eff.unsqueeze(1)

        # ---- Hover power baseline for this leg's mass -------------------
        P_hover_leg = cls.hover_power_w(mass_leg, k_thrust_leg)   # [B, seq]

        # ---- Travel energy: hover + drag term ---------------------------
        t_flight_s = dist_m / v_ms
        P_travel = P_hover_leg + k_drag_leg * (v_ms ** 3)
        E_travel_Wh = (P_travel / 3600.0) * t_flight_s

        # ---- Climb energy -------------------------------------------------
        E_climb_Wh = (mass_leg * cls.G_MS2 * climb_m) / (motor_eff_leg * 3600.0)

        # ---- Service / hover energy ----------------------------------------
        E_service_Wh = (P_hover_leg / 3600.0) * service_times

        leg_Wh = E_travel_Wh + E_climb_Wh + E_service_Wh   # [B, seq_len]
        total_Wh = leg_Wh.sum(dim=1)                        # [B]

        return total_Wh, leg_Wh

    # ------------------------------------------------------------------
    @classmethod
    def soc_drop_from_energy(cls, total_Wh, battery_Wh):
        """
        Optional convenience: convert energy (Wh) to SOC% drop, given
        each robot's own battery capacity in Wh (capacity_mAh * V / 1000).
        Kept separate so the core model stays energy-first.
        """
        return (total_Wh / battery_Wh) * 100.0


# ============================================================
# Example usage
# ============================================================
if __name__ == "__main__":
    B = 2          # batch of 2 robots/routes
    seq_len = 4    # 4 legs (e.g. depot -> t1 -> t2 -> t3 -> depot)

    # Route coordinates: [B, seq_len+1, 3], normalised [0,1]^3
    completed_coords = torch.tensor([
        [[0.0, 0.0, 0.05], [0.3, 0.2, 0.5], [0.5, 0.4, 0.6], [0.7, 0.6, 0.5], [0.0, 0.0, 0.05]],
        [[1.0, 1.0, 0.03], [0.8, 0.7, 0.4], [0.6, 0.5, 0.5], [0.4, 0.3, 0.4], [1.0, 1.0, 0.03]],
    ])

    speeds = torch.tensor([
        [1.0, 1.0, 0.8, 1.2],
        [0.9, 1.1, 1.0, 1.0],
    ])  # normalised speed per leg

    service_times = torch.tensor([
        [5.0, 6.0, 4.0, 0.0],   # last leg = return to depot, no service
        [5.0, 5.0, 5.0, 0.0],
    ])

    payload_drop_kg = torch.tensor([
        [2.0, 3.0, 1.0, 0.0],   # drops happen at legs 1-3, none on return leg
        [4.0, 2.0, 2.0, 0.0],
    ])

    robot_own_weight_kg = torch.tensor([5.0, 12.5])     # e.g. category 1 vs category 4
    initial_payload_kg = torch.tensor([6.0, 8.0])

    # Per-robot physical coefficients -- NOT hardcoded to any single
    # drone category; supply these from each category's own spec sheet.
    k_thrust = torch.tensor([0.012, 0.009])   # smaller frame -> less efficient -> higher k
    k_drag = torch.tensor([0.05, 0.08])
    motor_eff = torch.tensor([0.85, 0.88])

    total_Wh, leg_Wh = GeneralUAVEnergyModel.compute_batch(
        completed_coords, speeds, service_times, payload_drop_kg,
        robot_own_weight_kg, initial_payload_kg,
        k_thrust, k_drag, motor_eff,
    )

    print("Per-leg energy (Wh):\n", leg_Wh)
    print("Total energy per robot (Wh):\n", total_Wh)

    # Optional: convert to SOC% drop if you want it too
    battery_Wh = torch.tensor([88.8, 177.6])  # e.g. 8000mAh vs 16000mAh @ 11.1V
    soc_drop = GeneralUAVEnergyModel.soc_drop_from_energy(total_Wh, battery_Wh)
    print("SOC drop (%):\n", soc_drop)