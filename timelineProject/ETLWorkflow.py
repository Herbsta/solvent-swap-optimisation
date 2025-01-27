from manim import *
import numpy as np

class ETLWorkflow(Scene):
    def construct(self):
        # Title
        title = Text("Data Preparation: ETL Workflow", font_size=36)
        title.to_edge(UP)
        
        # Create source file icons
        def create_file_icon(file_type, color):
            file_box = Rectangle(height=1, width=0.8, color=color)
            extension = Text(file_type, font_size=20, color=color)
            extension.next_to(file_box, DOWN, buff=0.1)
            return VGroup(file_box, extension)
        
        # Create source files
        csv_file = create_file_icon("CSV", BLUE)
        excel_file = create_file_icon("XLSX", GREEN)
        
        # Position source files
        source_files = VGroup(csv_file, excel_file)
        source_files.arrange(DOWN, buff=1)
        source_files.to_edge(LEFT, buff=2)
        
        # Create SQLite database icon
        db_cylinder = Cylinder(radius=0.6, height=1.5, fill_color=GRAY, fill_opacity=0.7)
        db_label = Text("SQLite", font_size=24, color=GRAY)
        db_label.next_to(db_cylinder, DOWN, buff=0.1)
        database = VGroup(db_cylinder, db_label)
        database.to_edge(RIGHT, buff=2)
        
        # Create ETL process boxes
        process_boxes = VGroup()
        process_labels = ["Extract", "Transform", "Load"]
        
        for i, label in enumerate(process_labels):
            box = Rectangle(height=0.8, width=1.5, color=WHITE)
            text = Text(label, font_size=24)
            text.move_to(box)
            process = VGroup(box, text)
            process_boxes.add(process)
        
        process_boxes.arrange(RIGHT, buff=1)
        process_boxes.move_to(ORIGIN)
        
        # Create arrows between components
        arrows = VGroup()
        
        # Arrows from files to Extract
        for file in source_files:
            arrow = Arrow(
                file.get_right(),
                process_boxes[0].get_left(),
                buff=0.2,
                color=WHITE
            )
            arrows.add(arrow)
        
        # Arrows between process boxes
        for i in range(len(process_boxes) - 1):
            arrow = Arrow(
                process_boxes[i].get_right(),
                process_boxes[i + 1].get_left(),
                buff=0.2,
                color=WHITE
            )
            arrows.add(arrow)
        
        # Arrow to database
        final_arrow = Arrow(
            process_boxes[-1].get_right(),
            database.get_left(),
            buff=0.2,
            color=WHITE
        )
        arrows.add(final_arrow)
        
        # Animation sequence
        self.play(Write(title))
        self.wait(0.5)
        
        # Animate source files
        self.play(
            LaggedStartMap(Create, source_files),
            run_time=2
        )
        
        # Animate database
        self.play(Create(database))
        
        # Animate process boxes
        self.play(
            LaggedStartMap(Create, process_boxes),
            run_time=2
        )
        
        # Animate arrows
        self.play(
            LaggedStartMap(Create, arrows),
            run_time=2
        )
        
        # Add details for each process
        details = [
            # Extract details
            VGroup(
                Text("• Read source files", font_size=16),
                Text("• Validate format", font_size=16),
                Text("• Parse headers", font_size=16)
            ).arrange(DOWN, aligned_edge=LEFT).next_to(process_boxes[0], DOWN),
            
            # Transform details
            VGroup(
                Text("• Clean data", font_size=16),
                Text("• Standardize format", font_size=16),
                Text("• Handle missing values", font_size=16)
            ).arrange(DOWN, aligned_edge=LEFT).next_to(process_boxes[1], DOWN),
            
            # Load details
            VGroup(
                Text("• Create tables", font_size=16),
                Text("• Insert records", font_size=16),
                Text("• Verify integrity", font_size=16)
            ).arrange(DOWN, aligned_edge=LEFT).next_to(process_boxes[2], DOWN)
        ]
        
        # Animate details
        for detail_group in details:
            self.play(
                LaggedStartMap(Write, detail_group),
                run_time=1.5
            )
        
        # Add data flow animation
        def create_data_dot():
            return Dot(radius=0.1, color=YELLOW)
        
        # Animate data flowing through the pipeline
        for _ in range(2):  # Repeat animation twice
            # Create dots at source files
            dots = VGroup(*[create_data_dot().move_to(file.get_center()) 
                          for file in source_files])
            
            self.play(Create(dots))
            
            # Animate dots following the arrows to Extract
            self.play(
                *[dot.animate.move_to(process_boxes[0].get_center()) 
                  for dot in dots],
                run_time=1
            )
            
            # Combine dots
            combined_dot = create_data_dot().move_to(process_boxes[0].get_center())
            self.play(
                ReplacementTransform(dots, combined_dot)
            )
            
            # Move through transform and load
            for i in range(1, len(process_boxes)):
                self.play(
                    combined_dot.animate.move_to(process_boxes[i].get_center()),
                    run_time=1
                )
            
            # Move to database
            self.play(
                combined_dot.animate.move_to(database.get_center()),
                run_time=1
            )
            
            # Fade out dot
            self.play(FadeOut(combined_dot))
        
        self.wait(2)

if __name__ == "__main__":
    scene = ETLWorkflow()
    scene.render()