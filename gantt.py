import matplotlib.pyplot as plt
import numpy as np

# Define tasks with start and end months
tasks = [
    ("Conception of the idea and Title making", 0, 1, 'red'),
    ("Synopsis writing and discussion with supervisor", 1, 2, 'orange'),
    ("Presentation of synopsis and IRB approval", 2, 3, 'yellow'),
    ("Data collection and data entry in SPSS (Phase 1)", 3, 4, 'green'),
    ("Data collection and data entry in SPSS (Phase 2)", 4, 5, 'blue'),
    ("Data collection and data entry in SPSS (Phase 3)", 5, 6, 'purple'),
    ("Data analysis and completion of the document", 6, 7, 'brown'),
    ("Completion in making of PowerPoint presentation of Thesis", 7, 8, 'cyan'),
    ("Presentation of research report", 8, 9, 'pink')
]

fig, ax = plt.subplots(figsize=(12, 6))

# Generate y positions for each task
y_pos = np.arange(len(tasks))

# Plot the Gantt bars
for i, (task, start, end, color) in enumerate(tasks):
    ax.barh(y_pos[i], end - start, left=start, color=color, edgecolor='black', label=task if i == 0 else "_nolegend_")

# Add labels to bars
for i, (task, start, end, color) in enumerate(tasks):
    ax.text(start + 0.1, y_pos[i], task, va='center', fontsize=10, color='white', fontweight='bold')

# Set y-axis labels
ax.set_yticks(y_pos)
ax.set_yticklabels([task[0] for task in tasks], fontsize=10)

# Set x-axis labels (Months)
ax.set_xticks(np.arange(10))
ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct"], fontsize=10)
ax.set_xlabel("Months", fontsize=12)
ax.set_title("Project Timeline: Gantt Chart", fontsize=14, fontweight='bold')

# Add grid and legend
ax.grid(axis='x', linestyle='--', alpha=0.7)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1), fontsize=9)

# Display the chart
plt.tight_layout()
plt.show()
