# AICB: Analog Integrated Circuit Benchmark

[中文版說明](README_zh.md)

AICB (Analog Integrated Circuit Benchmark) is a specialized benchmark designed to evaluate the performance of large language models (LLMs) on tasks related to analog integrated circuit design, analysis, and understanding.

## Overview

Analog integrated circuit design requires deep domain knowledge across multiple disciplines including semiconductor physics, circuit theory, and electronic systems. This benchmark tests an LLM's capability to reason about analog circuit concepts, solve circuit problems, and demonstrate understanding of fundamental analog IC principles.

## Dataset Structure

The benchmark dataset consists of multiple-choice questions covering various domains of analog integrated circuit design. Each question has four options (A, B, C, D) with one correct answer.

The dataset is provided in CSV format with the following columns:

- `Question Number`: Unique identifier for each question
- `Question`: The question text
- `Option A`: First option
- `Option B`: Second option
- `Option C`: Third option
- `Option D`: Fourth option
- `Correct Answer`: The correct answer (A, B, C, or D)
- `Domain`: The specific analog IC domain the question belongs to

## Domains Covered

The benchmark covers several key domains in analog IC design:

- Amplifier Design (operational amplifiers, differential amplifiers)
- Noise Analysis
- Feedback Theory
- Frequency Response
- Stability and Compensation
- Biasing Techniques
- Bandgap References
- Analog-to-Digital and Digital-to-Analog Converters
- Oscillator Design
- Power Management Circuits
- Process, Voltage, and Temperature (PVT) Considerations

![Domain](./assets/domain_statistics.png)

## Usage

### Install

```bash
git clone https://github.com/yourusername/AICB-Analog-Integrated-Circuit-Benchmark.git
cd AICB-Analog-Integrated-Circuit-Benchmark
pip install pandas matplotlib
```

### Run Statistics

```bash
python domain_statistics.py
```

This will generate a statistical chart showing the distribution of questions across different domains.

## Output

The program generates:

- Console output showing question counts for each domain
- A high-resolution (300 DPI) chart saved as `domain_statistics.png`
- Optimized label display for better readability

## Citation

If you use AICB in your research, please cite:

```latex
@misc{aicb2025,
  author = {[Po-Hsiang, Hsu]},
  title = {AICB: Analog Integrated Circuit Benchmark},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{<https://github.com/treeleaves30760/AICB-Analog-Integrated-Circuit-Benchmark>}}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
