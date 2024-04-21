# Creative Generation with LLMs

**Team:** Gati Aher, Irmak Bukey, Isabel Suizo  
**Project Type:** Evaluation of Existing Models  
**Topic:** Mixture of Experts Models, particularly Mixtral-8x7B

In this project, we will evaluate the capabilities of recently released open-source models Mixtral 8x7B, Mixtral 8x7B–Instruct, and BlackMamba on an extended metaphor generation task and a rich summarization benchmark task. We will compare the performance of these new models to that of Llama2 and GPT-3.5 Turbo. Where possible, we will use existing datasets and benchmark metrics, or compare to an experimental set-up used in prior work. Our research could guide the choice of models for AI-enhanced human creativity tasks, offering potential benefits for business and education.

## Topic Overview

### MODELS

Mixtral 8x7B is a Sparse MoE language model with open weights released last December. The open-source community is very excited about Mixtral 8x7B because it outperformed Llama2 70B on most benchmarks with 6x faster inference, and the fine-tuned version Mixtral 8x7B–Instruct could match or outperform GPT3.5 on most standard benchmarks (Jiang et al., 2024). Related work in February combined cheap and fast inference from Mixture of Experts with the linear-complexity generation of Mamba State-Space Models and released the open-source BlackMamba model (Anthony et al., 2024).

Mixtral 8x7B, Mixtral 8x7B–Instruct, and BlackMamba are available on HuggingFace.
* Zyphra/BlackMamba-2.8B | Hugging Face
* Mixtral | Hugging Face
 
### TASK 1: Extended Metaphor Generation

Kim et al. (2023) were the first to focus on the constraints and challenges of generating extended metaphors. They ran user studies to study whether a few-shot GPT-3 (text-davinci-002) powered tool met science writer’s expectations for creating extended metaphors that could help a general audience easily understand scientific concepts. We will use Kim et al. (2023)’s [dataset of 600 extended metaphors](https://github.com/ucsd-creativitylab/metaphor-dataset) and rubric for classifying 5 LLM failure modes to evaluate Mixtral 8x7B, Mixtral 8x7B–Instruct, BlackMamba, and GPT3.5 on an extended metaphor creation task. 

In crowd-worker tasks, providing examples can create unintentional conformity effects. For an innovation-focused crowd-worker task, Yu et al. (2014) developed a distributed analogical idea generation process that helped make idea generation more effective and less reliant on chance. We will take inspiration from Yu et al. (2014) to create LLM prompts that support diversity and creativity in the extended metaphor creation task.

### TASK 2: Knowledge Specific Summarization

Retrieval-augmented generation (RAG) (Lewis et. al, 2020), is a technique for reducing LLM hallucinations in knowledge specific tasks. It involves integrating an LLM’s memory and capabilities (learned via pre-training) with a non-parametric external text memory (such as an index over Wikipedia pages) so that the LLM can query relevant documents while generating its answer. We will create RAG-versions of our models by combining a task prompt with a RAG prompt created by using LLM-generated keywords to find related Wikipedia articles via the [wikipedia package](https://pypi.org/project/wikipedia/).

We will evaluate our models and RAG models on Wikipedia summarization tasks from the USB benchmark (Kundan et al., 2023). We will score performance using ROUGE (Lin, 2004) against gold standards in the benchmark.

### References

Jiang, Albert Q., et al. "Mixtral of experts." arXiv preprint arXiv:2401.04088 (2024).

Anthony, Quentin, et al. "BlackMamba: Mixture of Experts for State-Space Models." arXiv preprint arXiv:2402.01771 (2024).

Zoph, Barret, et al. "St-moe: Designing stable and transferable sparse expert models." arXiv preprint arXiv:2202.08906 (2022).

Xue, Fuzhao, et al. "Openmoe: An early effort on open mixture-of-experts language models." arXiv preprint arXiv:2402.01739 (2024).

Kim, Jeongyeon, et al. "Metaphorian: Leveraging Large Language Models to Support Extended Metaphor Creation for Science Writing." Proceedings of the 2023 ACM Designing Interactive Systems Conference. 2023.

Yu, Lixiu, Aniket Kittur, and Robert E. Kraut. "Distributed analogical idea generation: inventing with crowds." Proceedings of the SIGCHI conference on Human Factors in Computing Systems. 2014.

Lewis, Patrick, et al. "Retrieval-augmented generation for knowledge-intensive nlp tasks." Advances in Neural Information Processing Systems 33. 2020.

Krishna, Kundan, et al. “USB: A Unified Summarization Benchmark Across Tasks and Domains”. arXiv [Cs.CL], 2023.

Chin-Yew Lin. 2004. ROUGE: A Package for Automatic Evaluation of Summaries. In Text Summarization Branches Out, pages 74–81, Barcelona, Spain. Association for Computational Linguistics. 2004.

