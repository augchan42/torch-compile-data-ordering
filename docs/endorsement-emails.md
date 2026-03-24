# arXiv Endorsement Outreach

Target category: **cs.LG** (Machine Learning)

## Priority Contacts

### 1. Yang Zhang — Ecole Polytechnique (BEST FIT — curriculum learning)

- **Paper**: "Beyond Random Sampling: Curriculum Learning" (arXiv:2506.11300)
- **Email**: yang.zhang@polytechnique.edu
- **Why**: Your paper directly builds on their finding. They showed curriculum learning works; you show it may be a compilation artifact.

**Draft email:**

> Subject: arXiv endorsement request — torch.compile amplifies data ordering effects (builds on your curriculum learning work)
>
> Dear Dr. Zhang,
>
> I'm writing about a finding that directly builds on your work in "Beyond Random Sampling" (arXiv:2506.11300). In controlled experiments on two platforms, I found that the curriculum learning effects you document may be partially explained by an interaction with torch.compile.
>
> Specifically: random shuffling of a 64-batch buffer improves val_bpb by 0.106–0.151 on CUDA with torch.compile (2.7–3.8× the seed noise floor), while the same shuffling produces improvements of only 0.008–0.037 on Apple MLX (no compilation step) — within measurement noise. The effect persists across both token diversity and compression ratio metrics.
>
> I believe this has important implications for the curriculum learning community: experiments conducted under torch.compile may report effects that don't generalize to other frameworks.
>
> The paper is 8 pages with 2 figures and 6 tables, based on 50+ controlled experiments. I'd be grateful if you would consider endorsing my first arXiv submission in cs.LG. I've attached the PDF.
>
> Paper repo: https://github.com/augchan42/torch-compile-data-ordering
>
> Thank you for your time,
> Augustin Chan

---

### 2. Hantian Ding — Amazon AWS (data packing → your mechanism)

- **Paper**: "Fewer Truncations Improve Language Modeling" (arXiv:2404.10830)
- **Email**: Try via Amazon Science profile or LinkedIn
- **Twitter**: @HantianDing (exists but no posts)
- **Why**: Your finding identifies the mechanism by which their best-fit packing creates problems under compilation.

**Draft email:**

> Subject: arXiv endorsement request — shuffling fixes ordering bias from best-fit packing under torch.compile
>
> Dear Dr. Ding,
>
> Your work on best-fit document packing ("Fewer Truncations Improve Language Modeling") documented that packing creates deterministic ordering biases. I've found that torch.compile amplifies these biases significantly.
>
> In controlled experiments, best-fit packed data produces 0.106–0.151 bpb worse results under torch.compile compared to shuffled data — an effect 2.7–3.8× the seed noise floor. The same shuffling has no effect on Apple MLX (no compilation). An early implementation bug where GPU tensor cloning broke torch.compile by +1.0 bpb provides direct mechanistic evidence.
>
> This suggests a practical fix for compiled training pipelines using best-fit packing: insert a shuffle buffer. I believe this is relevant to your research community.
>
> Would you consider endorsing my first arXiv submission in cs.LG? The paper is attached (8 pages, 2 figures, 6 tables, 50+ experiments).
>
> Paper repo: https://github.com/augchan42/torch-compile-data-ordering
>
> Thank you,
> Augustin Chan

---

### 3. Fangyuan Yu — Thoughtworks (memorization-compression theory)

- **Paper**: "Memorization-Compression Cycles" (arXiv:2505.08727)
- **Email**: fangyuan.yu@thoughtworks.com
- **Twitter**: @fangyuan_yu
- **GitHub**: fangyuan-ksgk
- **Why**: Your compilation hypothesis connects to their memorization-compression dynamics framework.

**Draft email:**

> Subject: arXiv endorsement — torch.compile may amplify memorization phase (connects to your memorization-compression work)
>
> Hi Fangyuan,
>
> Your work on memorization-compression cycles inspired part of my analysis. I've found that torch.compile creates kernel-level specialization to sequential data patterns that may amplify the memorization phase you describe.
>
> The evidence: shuffling a 64-batch buffer improves validation by 0.106–0.151 bpb on compiled CUDA but has no effect on non-compiled MLX — same model, same data, same eval. The compiled kernels appear to "memorize" the sequential data structure, reducing compression pressure.
>
> Would you be willing to endorse my first arXiv submission in cs.LG? Paper attached (8 pages).
>
> Repo: https://github.com/augchan42/torch-compile-data-ordering
>
> Best,
> Augustin Chan

---

### 4. Kairong Luo — Tsinghua University (LR decay + curriculum)

- **Paper**: "How LR Decay Wastes Your Best Data" (arXiv:2511.18903)
- **Email**: luokr24@mails.tsinghua.edu.cn
- **Corresponding author**: Kaifeng Lyu (IIIS, Tsinghua)
- **GitHub**: thu-yao-01-luo
- **Why**: Your experiments tested both standard warmdown and constant LR, directly following their recommendation.

**Draft email:**

> Subject: arXiv endorsement — compilation confound in curriculum learning (follows your LR decay finding)
>
> Dear Mr. Luo (or Prof. Lyu),
>
> Following your recommendation in "How LR Decay Wastes Your Best Data," I tested curriculum orderings under both standard warmdown and constant LR. I found a confound that may affect curriculum research more broadly: the effects are torch.compile-specific.
>
> Under torch.compile, any data reordering improves val_bpb by 0.106–0.151 (2.7–3.8× noise floor). On Apple MLX (no compilation), the same reordering produces 0.008–0.037 — within noise. This holds across both LR regimes and both difficulty metrics.
>
> Would you consider endorsing my first arXiv submission in cs.LG?
>
> Repo: https://github.com/augchan42/torch-compile-data-ordering
>
> Thank you,
> Augustin Chan

---

## Alternative Channels

If email doesn't work within a week:

1. **Twitter/X post** with Figure 1 (cross-platform bar chart) — tag @fangyuan_yu and relevant ML accounts
2. **r/MachineLearning** — post as "torch.compile amplifies data ordering effects" with key finding
3. **ML Discord servers** (EleutherAI, LAION) — share and ask for endorsement
4. **OpenReview** — post as preprint while waiting for arXiv endorsement

## arXiv Endorsement Process

1. Go to https://arxiv.org/auth/endorse
2. Enter endorser's email
3. arXiv emails them a link
4. They click the link (takes 5 minutes)
5. You can then submit to cs.LG
