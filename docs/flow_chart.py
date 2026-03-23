# make_fig_2_2_flow.py
import matplotlib.pyplot as plt

def box(ax, x, y, w, h, text, fontsize=9):
    rect = plt.Rectangle((x, y), w, h, fill=False, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fontsize)

def arrow(ax, x1, y1, x2, y2):
    ax.arrow(x1, y1, x2 - x1, y2 - y1,
             length_includes_head=True, head_width=0.12, head_length=0.25)

def main(out_png="Fig2_Training_and_Processing_Flow.png",
         out_svg="Fig2_Training_and_Processing_Flow.svg"):
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 20); ax.set_ylim(0, 12); ax.axis('off')

    w = 3.2; h = 1.2
    y1 = 9.8; x0 = 1.0

    # Row 1: Data -> Frames -> Annotation -> Split -> Augment
    steps1 = [
        ("Data collection\n(Shenton Park Farm)\nMulti-angle videos (10°–90°)", x0, y1),
        ("Frame extraction\n(begin/middle/end)", x0+4, y1),
        ("Manual annotation\n(Roboflow: crop/weed)", x0+8, y1),
        ("Dataset split\ntrain/val/test = 70/20/10", x0+12, y1),
        ("Augmentation\n(flip/rotate/brightness)", x0+16, y1),
    ]
    for text, x, y in steps1: box(ax, x, y, w, h, text)
    for i in range(len(steps1)-1):
        x, y = steps1[i][1], steps1[i][2]
        arrow(ax, x+w, y+h/2, steps1[i+1][1], y+h/2)

    # Row 2: Config -> Train -> Evaluate -> Select Best
    y2 = 6.4
    steps2 = [
        ("Model config\nYOLOv8n\nimg=640, bs=16,\nepochs=50\nAdamW, lr=0.001", x0, y2),
        ("Training\n(transfer learning\non COCO pretrain)", x0+4, y2),
        ("Evaluation on val\nmetrics: mAP, P, R", x0+8, y2),
        ("Select best weights\n(by mAP on val)", x0+12, y2),
    ]
    for text, x, y in steps2: box(ax, x, y, w, h, text)
    for i in range(len(steps2)-1):
        x, y = steps2[i][1], steps2[i][2]
        arrow(ax, x+w, y+h/2, steps2[i+1][1], y+h/2)

    # Row 3: Inference -> CSV -> KMeans/RANSAC -> RCM -> Plots
    y3 = 3.0
    steps3 = [
        ("Deployment\nInference on\nunseen frames", x0, y3),
        ("Export detections\nCSV (cls, bbox,\nconf, frame_id)", x0+4, y3),
        ("Crop row detection\nKMeans + RANSAC", x0+8, y3),
        ("Row Coordinate\nMapping (RCM)", x0+12, y3),
        ("Weed distribution\nplots\n(intra & inter)", x0+16, y3),
    ]
    for text, x, y in steps3: box(ax, x, y, w, h, text)
    for i in range(len(steps3)-1):
        x, y = steps3[i][1], steps3[i][2]
        arrow(ax, x+w, y+h/2, steps3[i+1][1], y+h/2)

    # Vertical hand-offs
    arrow(ax, x0+16 + w/2, y1, x0+16 + w/2, y2+h)  # Augment -> Config
    arrow(ax, x0+12 + w/2, y2, x0+12 + w/2, y3+h)  # Best weights -> Deployment

    ax.set_title("Section 2.2 – Model Training and Downstream Processing Flow", pad=10)
    plt.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")
    print(f"Saved: {out_png}\nSaved: {out_svg}")

if __name__ == "__main__":
    main()
