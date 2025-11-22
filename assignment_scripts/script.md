**Slide 1:**
"Traffic sign recognition faces a key challenge: balancing accuracy with computational efficiency for embedded devices. While modern networks achieve high accuracy, speed remains under-researched. This study asks: can the 30-year-old LeNet-5 still compete?"

**Slide 2:**
"Our methodology has three parts: First, we compare four architectures on GTSRB. Second, we fine-tune LeNet-5 to maximize performance. Third, we conduct ablation studies to validate each design choice."

**Slide 3:**
"We use GTSRB with over 50,000 images across 43 classes. Our metrics balance prediction quality and computational efficiency."

**Slide 4:**
"MiniVGG leads in accuracy at 98%, but LeNet-5 delivers exceptional speed—processing 120,000 images per second, four times faster than MiniVGG, with only 2% accuracy loss. This demonstrates a fundamental accuracy-speed trade-off."

**Slide 5:**
"Ablation studies validate LeNet-5's design. Data augmentation showed minimal improvement—GTSRB already has natural variation. Alternative kernels and activations dramatically degraded performance, confirming that the original 5×5 kernels and ReLU are optimal for this shallow architecture."

**Slide 6:**
"Fine-tuning experiments with label smoothing and optimizer adjustments revealed that LeNet-5 already operates near its performance ceiling. This validates the robustness of the original architecture."

**Slide 7:**
"MiniVGG's deeper architecture with sequential 3×3 blocks creates a robust feature hierarchy for complex patterns, but requires 8.6× more computation. LeNet-5's shallow design with 5×5 kernels minimizes overhead while maintaining adequate capacity—enabling 4× faster inference."

**Slide 8:**
"For real-time embedded systems with limited resources, LeNet-5 is ideal. For safety-critical applications with available compute, MiniVGG's superior accuracy justifies the cost."

**Slide 9:**
"Current limitations include daytime-only images and ideal conditions. Future work should address adverse weather, nighttime scenarios, and real-world testing across multiple datasets."

**Slide 10:**
"LeNet-5 achieves 96% accuracy at 120,000 images per second, proving its continued relevance. While MiniVGG is more accurate, LeNet-5's 4× speed advantage and 87% lower computational cost make it ideal for embedded real-time applications. The 30-year-old architecture remains remarkably competitive."
