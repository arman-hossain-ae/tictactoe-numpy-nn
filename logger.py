import numpy as np
import matplotlib.pyplot as plt

class TrainingLog:
    def __init__(self):
        self.history = {}

    def record(self, step, **metrics):
        """Record one row of metrics at a given step."""
        self.history.setdefault("step", []).append(step)
        for name, value in metrics.items():
            self.history.setdefault(name, []).append(value)

    def save(self, path="training_log.npz"):
        np.savez_compressed(path, **{k: np.array(v) for k, v in self.history.items()})
        print(f"Log saved to {path}")

    def load(self, path="training_log.npz"):
        data = np.load(path, allow_pickle=True)
        self.history = {k: data[k].tolist() for k in data.files}
        return self.history





class Visualizer:
    def __init__(self, window_title="Monitor"):
        plt.ion() 
        self.window_title = window_title
        self.fig = None
        self.axes = {}
        self.lines = {}
        
    def _initialize_plots(self, metric_names):
        """Internal helper to build the grid layout based on incoming keys."""
        num_metrics = len(metric_names)
        
        colors = ['#E53935', '#43A047', '#1E88E5', '#FB8C00', '#8E24AA', '#00ACC1']

        # Create a vertical stack of subplots equal to the number of metrics
        self.fig, ax_list = plt.subplots(num_metrics, 1, figsize=(8, 2 * num_metrics), sharex=True)
        self.fig.canvas.manager.set_window_title(self.window_title)
        
        # If there's only 1 metric, matplotlib returns a standalone axis instead of a list
        if num_metrics == 1:
            ax_list = [ax_list]
            
        # Map each metric name to its dedicated subplot axis and line object
        for idx, name in enumerate(metric_names):
            ax = ax_list[idx]
            line_color = colors[idx % len(colors)]
            self.axes[name] = ax
            
            # Create a line drawing element for this plot
            self.lines[name], = ax.plot([], [], color=line_color, linewidth=2, label=name.capitalize())
            ax.set_ylabel(name.capitalize())
            ax.grid(True)
            ax.legend(loc="upper left")
            
        # Label the absolute bottom axis
        self.axes[metric_names[-1]].set_xlabel("Episodes")
        plt.tight_layout()

    def update(self, history):
        """
        Accepts the history dictionary directly from your TrainingLog.
        Automatically updates all plotted lines.
        """
        if "step" not in history or len(history["step"]) == 0:
            return

        steps = history["step"]
        # Filter out 'step' so we only plot the true variables
        metric_names = [k for k in history.keys() if k != "step"]
        
        # Lazily initialize the visual window the first time data comes in
        if self.fig is None:
            self._initialize_plots(metric_names)
            
        # Update each active line with new historical points
        for name in metric_names:
            if name in self.lines:
                self.lines[name].set_data(steps, history[name])
                
                # Dynamically re-scale the axes boundaries for this metric
                self.axes[name].relim()
                self.axes[name].autoscale_view()
                
        # Flush the graphics pipeline to update the screen window
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def keep_open(self):
        """Call at the end of execution to prevent the window from disappearing."""
        plt.ioff()
        plt.show()
