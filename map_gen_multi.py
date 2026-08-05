import tkinter as tk
from tkinter import filedialog, messagebox
import json
import os

MAPS_DIR = "maps"


class MapGenerator:
    def __init__(self, root):
        self.root = root
        self.root.title("GEN AI UAV Swarm - Multi-Robot Map Generator")
        self.root.geometry("1050x850")
        self.root.configure(bg="#121212")

        self.map_width = 800
        self.map_height = 800

        # Obstacles are shared across all robots on this map
        self.obstacles = []

        # Current (editable) robot data
        self.start_pos = None
        self.goal_pos = None
        self.tasks = []

        # Drag support for editing loaded points
        self.drag_target = None  # ("start"/"goal"/"task", index_or_None)

        # History of previously saved robots on this map (drawn transparently)
        self.completed_robots = []  # list of dicts: start, goal, tasks

        self.robot_num = 1

        self.setup_ui()

    # ------------------------------------------------------------------
    def setup_ui(self):
        self.tool_frame = tk.Frame(self.root, width=220, bg="#1e1e1e", padx=10, pady=10)
        self.tool_frame.pack(side=tk.LEFT, fill=tk.Y)

        tk.Label(self.tool_frame, text="Select Mode", fg="#00FFFF", bg="#1e1e1e",
                 font=("Arial", 14, "bold")).pack(pady=10)

        self.mode = tk.StringVar(value="Start")
        modes = ["Start (Orange)", "Goal (Red)", "Task (Cyan)", "Move",
                 "Square", "Rectangle", "Circle", "U-Shape"]
        for m in modes:
            tk.Radiobutton(self.tool_frame, text=m, variable=self.mode, value=m.split()[0],
                           fg="#ffffff", bg="#1e1e1e", selectcolor="#333333",
                           font=("Arial", 12)).pack(anchor=tk.W, pady=3)

        tk.Label(self.tool_frame, text="Task Sequence (e.g. 2,1,3):", fg="#ffffff", bg="#1e1e1e",
                 font=("Arial", 10)).pack(pady=(15, 5), anchor=tk.W)
        self.seq_entry = tk.Entry(self.tool_frame, bg="#333333", fg="#ffffff",
                                   insertbackground="#00FFFF", font=("Arial", 12))
        self.seq_entry.pack(fill=tk.X)

        tk.Label(self.tool_frame, text="Map Number:", fg="#ffffff", bg="#1e1e1e",
                 font=("Arial", 10)).pack(pady=(15, 5), anchor=tk.W)
        self.map_num_entry = tk.Entry(self.tool_frame, bg="#333333", fg="#ffffff",
                                       insertbackground="#00FFFF", font=("Arial", 12))
        self.map_num_entry.insert(0, "1")
        self.map_num_entry.pack(fill=tk.X)

        tk.Label(self.tool_frame, text="Robot Number:", fg="#ffffff", bg="#1e1e1e",
                 font=("Arial", 10)).pack(pady=(10, 5), anchor=tk.W)
        self.robot_num_entry = tk.Entry(self.tool_frame, bg="#333333", fg="#ffffff",
                                         insertbackground="#00FFFF", font=("Arial", 12))
        self.robot_num_entry.insert(0, "1")
        self.robot_num_entry.pack(fill=tk.X)

        tk.Button(self.tool_frame, text="Clear Robot (keep map)", command=self.clear_robot,
                  bg="#444444", fg="#ffffff").pack(pady=(25, 5), fill=tk.X)
        tk.Button(self.tool_frame, text="Clear Entire Map", command=self.clear_all,
                  bg="#662222", fg="#ffffff").pack(pady=5, fill=tk.X)
        tk.Button(self.tool_frame, text="Load Robot JSON (edit)", command=self.load_robot_json,
                  bg="#333366", fg="#ffffff").pack(pady=(20, 5), fill=tk.X)
        tk.Button(self.tool_frame, text="Save Robot JSON", command=self.save_to_json,
                  bg="#FFA500", fg="#000000", font=("Arial", 12, "bold")).pack(pady=10, fill=tk.X)

        self.canvas = tk.Canvas(self.root, width=self.map_width, height=self.map_height,
                                 bg="#0a0a0a", highlightthickness=1, highlightbackground="#00FFFF")
        self.canvas.pack(side=tk.RIGHT, padx=10, pady=10)
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)

    # ------------------------------------------------------------------
    def on_canvas_click(self, event):
        x, y = event.x, event.y
        mode = self.mode.get()

        if mode == "Move":
            self.pick_drag_target(x, y)
            return

        if mode == "Start":
            self.start_pos = [x, y]
            self.redraw_current()
        elif mode == "Goal":
            self.goal_pos = [x, y]
            self.redraw_current()
        elif mode == "Task":
            self.tasks.append([x, y])
            self.redraw_current()
        elif mode == "Square":
            self.obstacles.append({"type": "rectangle", "position": [x, y], "width": 100, "height": 50})
            self.redraw_all()
        elif mode == "Rectangle":
            self.obstacles.append({"type": "rectangle", "position": [x, y], "width": 50, "height": 100})
            self.redraw_all()
        elif mode == "Circle":
            self.obstacles.append({"type": "circle", "position": [x, y], "radius": 40})
            self.redraw_all()
        elif mode == "U-Shape":
            self.obstacles.append({"type": "u_shape", "position": [x, y], "size": 120, "thickness": 20})
            self.redraw_all()

    def pick_drag_target(self, x, y, radius=10):
        if self.start_pos and abs(self.start_pos[0] - x) < radius and abs(self.start_pos[1] - y) < radius:
            self.drag_target = ("start", None)
            return
        if self.goal_pos and abs(self.goal_pos[0] - x) < radius and abs(self.goal_pos[1] - y) < radius:
            self.drag_target = ("goal", None)
            return
        for i, t in enumerate(self.tasks):
            if abs(t[0] - x) < radius and abs(t[1] - y) < radius:
                self.drag_target = ("task", i)
                return
        self.drag_target = None

    def on_canvas_drag(self, event):
        if self.mode.get() != "Move" or self.drag_target is None:
            return
        x, y = event.x, event.y
        kind, idx = self.drag_target
        if kind == "start":
            self.start_pos = [x, y]
        elif kind == "goal":
            self.goal_pos = [x, y]
        elif kind == "task":
            self.tasks[idx] = [x, y]
        self.redraw_current()

    def on_canvas_release(self, event):
        self.drag_target = None

    # ------------------------------------------------------------------
    def redraw_all(self):
        """Full redraw: obstacles, faded previous robots, current robot."""
        self.canvas.delete("all")
        self.draw_obstacles()
        self.draw_completed_robots()
        self.draw_current_robot()

    def redraw_current(self):
        """Cheaper redraw when only current robot's points changed."""
        self.canvas.delete("current")
        self.draw_current_robot()

    def draw_obstacles(self):
        for obs in self.obstacles:
            x, y = obs["position"]
            if obs["type"] == "rectangle":
                w, h = obs["width"], obs["height"]
                self.canvas.create_rectangle(x - w/2, y - h/2, x + w/2, y + h/2,
                                              fill="#333333", outline="#00FFFF", tags="obstacle")
            elif obs["type"] == "circle":
                r = obs["radius"]
                self.canvas.create_oval(x - r, y - r, x + r, y + r,
                                         fill="#333333", outline="#00FFFF", tags="obstacle")
            elif obs["type"] == "u_shape":
                size, t = obs["size"], obs["thickness"]
                h = size / 2
                self.canvas.create_rectangle(x - h, y - h, x - h + t, y + h,
                                              fill="#333333", outline="#00FFFF", tags="obstacle")
                self.canvas.create_rectangle(x + h - t, y - h, x + h, y + h,
                                              fill="#333333", outline="#00FFFF", tags="obstacle")
                self.canvas.create_rectangle(x - h, y + h - t, x + h, y + h,
                                              fill="#333333", outline="#00FFFF", tags="obstacle")

    def draw_completed_robots(self):
        """Draw previously saved robots faded (stippled) so new points are
        clearly distinguishable and won't be mistaken as overlapping."""
        for robot in self.completed_robots:
            if robot.get("start"):
                x, y = robot["start"]
                self.canvas.create_oval(x - 6, y - 6, x + 6, y + 6, outline="#FFA500",
                                         stipple="gray50", fill="", width=2, tags="faded")
            if robot.get("goal"):
                x, y = robot["goal"]
                self.canvas.create_oval(x - 6, y - 6, x + 6, y + 6, outline="#FF0000",
                                         stipple="gray50", fill="", width=2, tags="faded")
            for i, t in enumerate(robot.get("tasks", [])):
                x, y = t
                self.canvas.create_oval(x - 6, y - 6, x + 6, y + 6, outline="#00FFFF",
                                         stipple="gray50", fill="", width=2, tags="faded")
                self.canvas.create_text(x + 12, y - 12, text=f"R{robot['robot_num']}-{i+1}",
                                         fill="#557777", font=("Arial", 9), tags="faded")

    def draw_current_robot(self):
        if self.start_pos:
            x, y = self.start_pos
            self.canvas.create_oval(x - 6, y - 6, x + 6, y + 6, fill="#FFA500", tags="current")
        if self.goal_pos:
            x, y = self.goal_pos
            self.canvas.create_oval(x - 6, y - 6, x + 6, y + 6, fill="#FF0000", tags="current")
        for i, t in enumerate(self.tasks):
            x, y = t
            self.canvas.create_oval(x - 6, y - 6, x + 6, y + 6, fill="#00FFFF", tags="current")
            self.canvas.create_text(x + 12, y - 12, text=str(i + 1), fill="#00FFFF",
                                     font=("Arial", 12, "bold"), tags="current")

    # ------------------------------------------------------------------
    def clear_robot(self):
        """Clear current robot's start/goal/tasks; obstacles and previously
        saved robots (shown transparently) remain on the map."""
        self.start_pos = None
        self.goal_pos = None
        self.tasks = []
        self.seq_entry.delete(0, tk.END)
        self.redraw_all()

    def clear_all(self):
        if not messagebox.askyesno("Confirm", "Clear the entire map including obstacles and history?"):
            return
        self.obstacles = []
        self.completed_robots = []
        self.start_pos = None
        self.goal_pos = None
        self.tasks = []
        self.seq_entry.delete(0, tk.END)
        self.canvas.delete("all")

    # ------------------------------------------------------------------
    def load_robot_json(self):
        path = filedialog.askopenfilename(initialdir=MAPS_DIR, filetypes=[("JSON files", "*.json")])
        if not path:
            return
        with open(path, "r") as f:
            data = json.load(f)

        self.obstacles = data.get("obstacles", [])
        self.start_pos = data.get("start_position")
        self.goal_pos = data.get("goal_position")
        task_points = data.get("task_points", {})
        # preserve order by key
        self.tasks = [task_points[k] for k in sorted(task_points, key=lambda v: int(v))]

        seq = data.get("task_sequence", [])
        self.seq_entry.delete(0, tk.END)
        self.seq_entry.insert(0, ",".join(str(s) for s in seq))

        meta = data.get("robot_metadata", {})
        self.map_num_entry.delete(0, tk.END)
        self.map_num_entry.insert(0, str(meta.get("map_number", self.map_num_entry.get())))
        self.robot_num_entry.delete(0, tk.END)
        self.robot_num_entry.insert(0, str(meta.get("robot_number", self.robot_num_entry.get())))

        self.redraw_all()
        messagebox.showinfo("Loaded", f"Loaded {os.path.basename(path)}. Switch to 'Move' mode to drag points.")

    # ------------------------------------------------------------------
    def save_to_json(self):
        if not self.start_pos:
            messagebox.showerror("Error", "Please set a Start position before saving.")
            return
        if not self.goal_pos:
            messagebox.showerror("Error", "Please set a Goal position before saving.")
            return

        seq_str = self.seq_entry.get().strip()
        if not seq_str:
            task_sequence = list(range(1, len(self.tasks) + 1))
        else:
            try:
                task_sequence = [int(s.strip()) for s in seq_str.split(",")]
            except ValueError:
                messagebox.showwarning("Warning", "Invalid sequence format. Defaulting to ascending order.")
                task_sequence = list(range(1, len(self.tasks) + 1))

        try:
            map_num = int(self.map_num_entry.get().strip())
            robot_num = int(self.robot_num_entry.get().strip())
        except ValueError:
            messagebox.showerror("Error", "Map Number and Robot Number must be integers.")
            return

        map_data = {
            "robot_metadata": {
                "map_number": map_num,
                "robot_number": robot_num,
                "size": [self.map_width, self.map_height],
                "num_obstacles": len(self.obstacles),
                "num_tasks": len(self.tasks)
            },
            "start_position": self.start_pos,
            "goal_position": self.goal_pos,
            "task_points": {i + 1: pos for i, pos in enumerate(self.tasks)},
            "task_sequence": task_sequence,
            "obstacles": self.obstacles
        }

        os.makedirs(MAPS_DIR, exist_ok=True)
        filename = os.path.join(MAPS_DIR, f"map_{map_num:03d}_robot_{robot_num}.json")
        with open(filename, "w") as f:
            json.dump(map_data, f, indent=4)

        # push this robot into history (faded) and prep for next robot
        self.completed_robots.append({
            "robot_num": robot_num,
            "start": self.start_pos,
            "goal": self.goal_pos,
            "tasks": list(self.tasks)
        })

        messagebox.showinfo("Saved", f"Saved to {os.path.abspath(filename)}")

        # auto-clear current robot & bump robot number for convenience
        self.robot_num_entry.delete(0, tk.END)
        self.robot_num_entry.insert(0, str(robot_num + 1))
        self.clear_robot()


if __name__ == "__main__":
    root = tk.Tk()
    app = MapGenerator(root)
    root.mainloop()
