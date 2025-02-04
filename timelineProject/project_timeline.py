from manim import *
from manim_slides import Slide

class SimpleTimeline(Slide):
    def construct(self):
        # Create the main timeline
        timeline = Line(
            start=LEFT * 5,
            end=RIGHT * 5,
            stroke_width=4,
            color=BLUE_B
        )
        
        # Waypoints
        waypoints = [
            "Extract, Transform & Load",
            "Feature Engineering",
            "Variational Autoencoder",
            "VAE Validation",
            "Bayesian Optimisation"
        ]
        
        # Create dots
        dots = VGroup(*[
            Dot(timeline.point_from_proportion(i/4))
            for i in range(5)
        ])
        
        # Create labels
        labels = VGroup(*[
            Text(waypoint, font_size=24)
            for waypoint in waypoints
        ])
        
        # Position labels alternately above and below
        for i, label in enumerate(labels):
            if i % 2 == 0:
                label.next_to(dots[i], UP * 1.5)
            else:
                label.next_to(dots[i], DOWN * 1.5)
        
        # Create everything at once
        self.play(
            Create(timeline),
            Create(dots),
            Write(labels)
        )