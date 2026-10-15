.PHONY: demo-arc demo-shakespeare demo-evolution demo-clip demo-diffusion demo-multimodal test eval-cpu eval-moe clean

# Run the ARC training demo
demo-arc:
	@echo "🚀 Running ARC Training Demo..."
	python3 -m nano_moe.train_arc

# Run the Tiny Shakespeare training demo
demo-shakespeare:
	@echo "🎭 Running Tiny Shakespeare Training Demo..."
	python3 -m nano_moe.main

# Run the Evolutionary Strategies demo
demo-evolution:
	@echo "🧬 Running Evolutionary Strategies Demo..."
	python3 examples/demo_evolution.py

# Run the CLIP training demo (vision-language)
demo-clip:
	@echo "🖼️ Running CLIP Training Demo..."
	python3 -m nano_moe.train_clip

# Run the Diffusion training demo (text-to-image)
demo-diffusion:
	@echo "🎨 Running Diffusion Training Demo..."
	python3 -m nano_moe.train_diffusion

# Run the Multimodal joint training demo (CLIP + Diffusion + Captioning + LM)
demo-multimodal:
	@echo "🌐 Running Multimodal Training Demo..."
	python3 -m nano_moe.train_multimodal

# Run the Phase Transformer Enhancements demo (Hybrid Wavelet, Adaptive Sparsity, FourierCLIP)
demo-phase:
	@echo "🌊 Running Phase Transformer Enhancements Demo..."
	python3 examples/demo_phase_enhancements.py

# Benchmark SFPT vs Standard Transformer
benchmark:
	@echo "🏆 Running SFPT vs Transformer Benchmark..."
	python3 benchmarks/benchmark_sfpt_vs_transformer.py

# Unit tests + short offline end-to-end runs
test:
	python3 -m pytest tests -q

# CPU eval: dense vs. MoE char-level LM on input.txt (~2 min)
eval-cpu:
	python3 evals/cpu_eval.py

# CPU eval: IntegratedMoE learning and routing on synthetic images (~1 min)
eval-moe:
	python3 evals/image_moe_eval.py

# Clean up training outputs
clean:
	@echo "🧹 Cleaning up..."
	rm -rf training_outputs/
	rm -rf integrated_results/
	rm -rf results/
	rm -rf sfpt_tinystories/
	rm -rf true_phase_text/
	rm -f *.log
