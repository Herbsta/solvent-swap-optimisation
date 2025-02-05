from manim import *
from manim_slides import Slide, ThreeDSlide

class SupervisorRequirements(Slide):
    def construct(self):
        # Title slide
        title = Text("Key Project Requirements", font_size=48)
        subtitle = Text("Solvent Selection Using ML", font_size=36).next_to(title, DOWN)
        
        self.play(Write(title), Write(subtitle))
        self.next_slide()
        
        # Timeline Overview
        self.play(FadeOut(title), FadeOut(subtitle))
        
        timeline = VGroup(
            Text("Data Collection", font_size=36),
            Text("ML Model Development", font_size=36),
            Text("Results Analysis", font_size=36),
            Text("Report Writing", font_size=36)
        ).arrange(DOWN, buff=1)
        
        arrows = VGroup(*[
            Arrow(timeline[i].get_right(), timeline[i+1].get_right(), color=BLUE)
            for i in range(len(timeline)-1)
        ])
        
        for item in timeline:
            self.play(Write(item))
        for arrow in arrows:
            self.play(Create(arrow))
            
        self.next_slide()
        
        # Objectives
        self.play(FadeOut(timeline), FadeOut(arrows))
        
        objectives_title = Text("Project Objectives", font_size=40).to_edge(UP)
        objectives = VGroup(
            Text("• Analyze solvent swapping methods", font_size=32),
            Text("• Develop ML model for solubility prediction", font_size=32),
            Text("• Evaluate green solvent alternatives", font_size=32),
            Text("• Compare with existing approaches", font_size=32)
        ).arrange(DOWN, aligned_edge=LEFT).next_to(objectives_title, DOWN)
        
        self.play(Write(objectives_title))
        self.play(Create(objectives))
        self.next_slide()
        
        # Deliverables
        self.play(FadeOut(objectives_title), FadeOut(objectives))
        
        deliverables_title = Text("Key Deliverables", font_size=40).to_edge(UP)
        deliverables = VGroup(
            Circle(radius=0.4, color=BLUE).set_fill(BLUE, opacity=0.5),
            Circle(radius=0.4, color=BLUE).set_fill(BLUE, opacity=0.5),
            Circle(radius=0.4, color=BLUE).set_fill(BLUE, opacity=0.5)
        ).arrange(RIGHT, buff=2)
        
        labels = VGroup(
            Text("Optimized\nDataset", font_size=24),
            Text("ML\nModel", font_size=24),
            Text("Technical\nPaper", font_size=24)
        )
        
        for circle, label in zip(deliverables, labels):
            label.next_to(circle, DOWN)
        
        self.play(Write(deliverables_title))
        for circle, label in zip(deliverables, labels):
            self.play(Create(circle), Write(label))
            
        self.next_slide()
        
        # Meeting Schedule
        self.play(
            FadeOut(deliverables_title), 
            FadeOut(deliverables), 
            FadeOut(labels)
        )
        
        meetings_title = Text("Supervision Schedule", font_size=40).to_edge(UP)
        calendar = Rectangle(height=3, width=4)
        calendar_text = Text("Bi-weekly Meetings", font_size=32).next_to(calendar, DOWN)
        
        points = VGroup(
            Text("• Progress updates", font_size=24),
            Text("• Technical discussion", font_size=24),
            Text("• Review of results", font_size=24)
        ).arrange(DOWN, aligned_edge=LEFT).next_to(calendar_text, DOWN)
        
        self.play(Write(meetings_title))
        self.play(Create(calendar), Write(calendar_text))
        self.play(Create(points))
        self.next_slide()
        
        # Final Reminder
        self.play(
            FadeOut(meetings_title), 
            FadeOut(calendar), 
            FadeOut(calendar_text), 
            FadeOut(points)
        )
        
        final_text = Text("Technical Paper Due: May 15", font_size=48)
        submission_note = Text("10 pages, journal format", font_size=36).next_to(final_text, DOWN)
        
        self.play(Write(final_text))
        self.play(Write(submission_note))
