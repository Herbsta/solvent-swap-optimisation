from manim import *
import numpy as np

class ProjectTimeline(Scene):
    def construct(self):
        # Timeline creation
        timeline = NumberLine(
            x_range=[0, 6, 1],
            length=10,
            color=BLUE,
            include_numbers=False,
        ).shift(UP * 2)
        
        # Timeline phases
        phases = [
            "Literature\nReview\nNov-Dec '24",
            "Data Collection\nJan-Feb '25",
            "ML Phase 1\nFeb-Mar '25",
            "ML Phase 2\nMar-Apr '25",
            "Report\nApr-May '25",
            "Final\nSubmission\nMay '25"
        ]
        
        # Create dots and labels for timeline
        dots = VGroup()
        labels = VGroup()
        
        for i, phase in enumerate(phases):
            dot = Dot(point=timeline.n2p(i), color=BLUE)
            label = Text(phase, font_size=20).next_to(dot, DOWN, buff=0.5)
            dots.add(dot)
            labels.add(label)
        
        # ML Workflow components
        ml_title = Text("Machine Learning Workflow", font_size=30)
        ml_title.move_to(UP * 0.5)
        
        # Create boxes for ML stages
        stages = ["Data Preparation", "Feature Engineering", "Model Development"]
        boxes = VGroup()
        stage_texts = VGroup()
        
        for i, stage in enumerate(stages):
            box = Rectangle(height=2, width=3, color=WHITE)
            box.shift(DOWN * 1.5 + RIGHT * (i * 4 - 4))
            text = Text(stage, font_size=20).move_to(box)
            boxes.add(box)
            stage_texts.add(text)
        
        # Arrows between boxes
        arrows = VGroup()
        for i in range(len(boxes) - 1):
            arrow = Arrow(
                boxes[i].get_right(),
                boxes[i + 1].get_left(),
                buff=0.1,
                color=WHITE
            )
            arrows.add(arrow)
        
        # Animation sequence
        self.play(Create(timeline))
        self.play(
            LaggedStartMap(Create, dots),
            LaggedStartMap(Write, labels),
            run_time=3
        )
        self.wait()
        
        # Fade out timeline and bring in ML workflow
        self.play(
            FadeOut(timeline),
            FadeOut(dots),
            FadeOut(labels),
            Write(ml_title)
        )
        
        # Create ML workflow diagram
        self.play(
            Create(boxes),
            Write(stage_texts),
            run_time=2
        )
        self.play(Create(arrows))
        
        # Add details for each stage
        details = [
            # Data Preparation details
            VGroup(
                Text("• Dataset compilation", font_size=16),
                Text("• Standardization", font_size=16),
                Text("• Quality checks", font_size=16)
            ).arrange(DOWN, aligned_edge=LEFT).next_to(boxes[0], DOWN),
            
            # Feature Engineering details
            VGroup(
                Text("• SMILES encoding", font_size=16),
                Text("• Physical properties", font_size=16),
                Text("• Feature selection", font_size=16)
            ).arrange(DOWN, aligned_edge=LEFT).next_to(boxes[1], DOWN),
            
            # Model Development details
            VGroup(
                Text("• Model selection", font_size=16),
                Text("• Training & Testing", font_size=16),
                Text("• Performance evaluation", font_size=16)
            ).arrange(DOWN, aligned_edge=LEFT).next_to(boxes[2], DOWN)
        ]
        
        # Animate details
        for detail_group in details:
            self.play(
                LaggedStartMap(Write, detail_group),
                run_time=2
            )
        
        self.wait(2)

if __name__ == "__main__":
    scene = ProjectTimeline()
    scene.render()