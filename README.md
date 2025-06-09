<h1 align="center"> <p>Parameter-Efficient Fine-Tuning of State Space Models</p></h1>
<h4 align="center">
    <p>
      <a href="https://scholar.google.com/citations?user=G1EpeWYAAAAJ&hl=en" target="_blank">Kevin Galim</a><sup>*1</sup>, <a href="https://scholar.google.com/citations?user=Q-ARWkwAAAAJ&hl=eh" target="_blank">Wonjun Kang</a><sup>*1</sup>, <a href="https://yzeng58.github.io/zyc_cv/" target="_blank">Yuchen Zeng</a><sup>*2</sup>, <a href="http://cvml.ajou.ac.kr/wiki/index.php/Professor" target="_blank">Hyung Il Koo</a><sup>1</sup>, <a href="https://kangwooklee.com/aboutme/" target="_blank">Kangwook Lee</a><sup>2</sup>
  </p>
  <p>
    <sup>1</sup> FuriosaAI, <sup>2</sup> UW-Madison
   </p>
    </h4>
<p align="center">
    <a href="https://arxiv.org/abs/2410.09016">
        <img alt="GitHub release" src="https://img.shields.io/badge/arXiv-2410.09016-b31b1b.svg">
    </a>
</p>

<p align="center">
<img src = "https://github.com/user-attachments/assets/b318c473-cb41-4d88-9e83-d9e1ac03b620" width="30%" height="30%">
</p>

**Abstract**: Deep State Space Models (SSMs), such as Mamba (Gu & Dao, 2024), have become powerful tools for language modeling, offering high performance and linear scalability with sequence length. However, the application of parameter-efficient fine-tuning (PEFT) methods to SSM-based models remains largely underexplored. We start by investigating two fundamental questions on existing PEFT methods: (i) How do they perform on SSM-based models? (ii) Which parameters should they target for optimal results? Our analysis shows that LoRA and its variants consistently outperform all other PEFT methods. While LoRA is effective for linear projection matrices, it fails on SSM modules—yet still outperforms other methods applicable to SSMs, indicating their limitations. This underscores the need for a specialized SSM tuning approach. To address this, we propose Sparse Dimension Tuning (SDT), a PEFT method tailored for SSM modules. Combining SDT for SSMs with LoRA for linear projection matrices, we achieve state-of-the-art performance across extensive experiments.

# News  🚀

- `2025-05-01` Our paper has been accepted to <a href="https://icml.cc/virtual/2025/poster/46398">ICML 2025</a>! 🎉🎉🎉
- `2024-11-01` Our paper is selected for oral presentation (5 of 92 accepted papers) at <a href="https://sites.google.com/view/neurips2024-ftw/home">NeurIPS 2024 Workshop FITML</a>! 🎉🎉
- `2024-10-11` Our paper is available on <a href="https://arxiv.org/abs/2410.09016">arXiv</a>!
- `2024-10-09` Our paper has been accepted to <a href="https://sites.google.com/view/neurips2024-ftw/home">NeurIPS 2024 Workshop FITML</a>! 🎉


# Usage

* PEFT implementation on **S4**: Refer to the [S4 folder](./S4).
* PEFT implementation on **Mamba**: Refer to the [mamba-peft folder](./mamba-peft).

# Citation
```tex
@article{galim2024parameter,
  title={Parameter-Efficient Fine-Tuning of State Space Models},
  author={Galim, Kevin and Kang, Wonjun and Zeng, Yuchen and Koo, Hyung Il and Lee, Kangwook},
  journal={arXiv preprint arXiv:2410.09016},
  year={2024}
}
```
