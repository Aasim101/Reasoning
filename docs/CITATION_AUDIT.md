# Citation audit

Audit of every entry in `paper/references.bib`, completed 29 July 2026.  
**Reframing pass (same date):** added 8 verified entries for BrittleBench/TEE line; verified Bay & Yearick (arXiv:2606.28661); flagged `wang2025measuring` unresolved.

**Outcome: 102 entries** (94 prior + 8 new foundations), **101 verified**, **1 unresolved external reference** (`wang2025measuring` — not in bib; see §8). **GPU tier totals recomputed 29 Jul 2026:** Tier A ≈ **12.5 h**, Tier B ≈ **16.1 h**, paper-critical ≈ **28.6 h** (~**32 h** with pilots/slack); core factor = 3 levels (c0–c2). O2 = 1.75 h (17.6 M tok); O2b = 0.5 h (5.28 M tok). **arXiv API re-verified this pass:** 2603.13285, 2604.11581, 2606.19636, 2607.13304, 2603.01865, 2405.17202, 2606.28661 — all resolve.

Cross-check against `main.tex`: all `\cite` keys in prose must resolve; new keys:
`brittlebench2026`, `tee2026`, `hardunreached2026`, `facetnoise2026`, `cyclicjudge2026`,
`prompteval2024`, `mizrahi2024multiprompt`, `alzahrani2024leaderboard`.

---

## 8. July 2026 reframing additions (post-adversarial review)

| Key | Status | Source | Notes |
|---|---|---|---|
| `brittlebench2026` | **V-ARXIV** | arXiv API 2603.13285 | 11 authors; title canonicalised to BrittleBench casing |
| `tee2026` | **V-ARXIV** | arXiv API 2604.11581 | Solomon Messing; TEE / crossed REML |
| `hardunreached2026` | **V-ARXIV** | arXiv API 2606.19636 | Zhou et al.; sampling blind spot |
| `facetnoise2026` | **V-ARXIV** | arXiv API 2607.13304 | Żatuchin; G-theory brand answers |
| `cyclicjudge2026` | **V-ARXIV** | arXiv API 2603.01865 | Zhu et al.; judge variance decomposition |
| `prompteval2024` | **V-ARXIV** | arXiv API 2405.17202 | NeurIPS 2024 per arXiv comment; 9 authors |
| `mizrahi2024multiprompt` | **V-PUB** | ACL Anthology 2024.tacl-1.52 | TACL vol. 12, pp. 933--949 |
| `alzahrani2024leaderboard` | **V-ACL** | Anthology 2024.acl-long.744.bib | 12 authors; pp. 13787--13805 |

### Re-verified this pass

| Key | Finding |
|---|---|
| `bay2026modal` | **Exists.** arXiv:2606.28661; Bay & Yearick; preprint; code at github.com/bay-yearick-lab/sampling-ceilings. Load-bearing use reduced to one gap citation in reframed prose; π_mode defined self-containedly in Method. |

### Unresolved (do not cite until closed)

| Reference | Status |
|---|---|
| **`wang2025measuring`** | BrittleBench body cites this key for prior variance-decomposition work. **Not resolved.** Candidate arXiv:2512.21326 (Sida Wang, *Measuring all the noises of LLM Evals*) is a **different** decomposition (prediction vs. data noise, not item vs. prompt). Fetch BrittleBench author `.bib` or contact authors before adding. **Not added to `references.bib`.** |

---

## 1. Headline result (updated counts)

| Outcome | Count |
|---|---|
| Verified, no change needed | 87 |
| Verified with a correction applied (prior pass) | 7 |
| **New entries (reframing pass)** | **8** |
| Unresolved external (not in bib) | 1 (`wang2025measuring`) |
| **Total in `references.bib`** | **102** |

At the start of this pass, 28 entries had no `author` field at all and carried
an `AUTHOR LIST UNVERIFIED` note. **All 28 are now resolved from an
authoritative source.** This mattered mechanically as well as scholastically:
`natbib` renders `\citet{key}` for an author-less entry as a malformed label,
so 15 of the `\citet{...}` calls in the Related Work section would have
produced visible garbage in the compiled PDF.

## 2. The argumentatively load-bearing citations

These six were checked first in the original pass; **reframing adds BrittleBench/TEE as foundations, not competitors.**

| Key | Verified as | Reframing note |
|---|---|---|
| `bay2026modal` | Bay & Yearick, arXiv:2606.28661 | Re-verified 29 Jul 2026. Reduced to gap citation; π_mode defined in-paper. |
| `brittlebench2026` | Romanou et al., arXiv:2603.13285 | **Foundation** — prompt share ≈50%; deterministic scoring. |
| `tee2026` | Messing, arXiv:2604.11581 | **Foundation** — REML item×prompt; aggregate estimand contrast. |
| `sclar2024formatspread` | Sclar et al., ICLR 2024 | Aggregate format sensitivity; delta is modes + stochastic CoT. |
| `prompteval2024` | Polo et al., NeurIPS 2024 | IRT over item×template; psychometric antecedent. |
| `gema2025inverse` | Gema et al., TMLR 2025 | Unchanged. |

## 3. Corrections applied

| Key | What was wrong | Corrected to | Source of truth |
|---|---|---|---|
| `rcs2026` | **Title was wrong.** The bib carried *"Radial Consensus Score for Answer Aggregation in Language Model Reasoning"*, a placeholder I had reconstructed from the PDF body because search had surfaced only the PDF. It was explicitly flagged `TITLE AND AUTHORS UNVERIFIED`. | *"Beyond Majority Voting: Efficient Best-Of-N with Radial Consensus Score"*; Manh Nguyen, Sunil Gupta, Hung Le | arXiv API, 2604.12196 |
| `weightedreasoning2024` | **Title was stale.** The bib carried the v1 title *"Enhancing Language Model Reasoning via Weighted Reasoning in Self-Consistency"*. | *"Semantic Self-Consistency: Enhancing Language Model Reasoning via Semantic Weighting"*; Knappe, Li, Chauhan, Chhua, Zhu, O'Brien | arXiv API, 2410.07839 |
| `precotprobe2026` | **Title was stale.** The bib carried a v1 title read off the PDF. | *"Post-Hoc Reasoning in Chain of Thought: Decoding and Steering Pre-Committed Answers"*; Cox, Kianersi, Garriga-Alonso | arXiv API, 2603.01437 |
| `zhou2024paraphrase` | **First author's given name was wrong.** The bib had *"Zhou, Yuting and others"*, inferred from a citing paper. | Yue Zhou, Yada Zhu, Diego Antognini, Yoon Kim, Yang Zhang; NAACL 2024, pp. 2793--2804 | ACL Anthology BibTeX |
| `veccisc2026` | Fourth author was truncated to `others`. | Nianwen Xue | ACL Anthology BibTeX |
| `sclar2024formatspread` | Spurious comma in the title (*"...Prompt Design, or: How I learned..."*). | No comma before `or:` | The paper's own title block |
| `li2024esc` | Carried a note saying the venue was uncertain between ICLR 2024 and ACL 2024. | ICLR 2024, confirmed; note removed | `iclr.cc/virtual/2024/poster/17848` and `proceedings.iclr.cc` |

## 4. Things a reviewer should know, reported rather than patched

Three findings are worth flagging explicitly, because they affect how much
weight the prose can put on a source. None of them is a misattribution.

**`bay2026modal` is a two-author preprint, not a peer-reviewed paper.** Both
authors are marked equal contribution, the affiliation line reads "PhD,
University of Illinois at Urbana-Champaign", and correspondence is to gmail
addresses. The paper exists and says exactly what we attribute to it — the
modal ceiling, the correlation ceiling, the identifiability gap — but it has
not been through review. Since our framing leans on it in the abstract, the
introduction, the related work and the method, the bib entry now carries an
explicit `note` recording its preprint status. The paper's argument does not
*depend* on the modal-ceiling result being correct (we measure $\pi_{\text{mode}}$
ourselves), but the framing does depend on it being a real and citable object,
which it is.

**`sclar2024formatspread` is closer to our claim than the gap analysis
originally allowed.** In verifying the title I read the abstract in full, and
it states that "format performance only weakly correlates between models, which
puts into question the methodological validity of comparing models with an
arbitrarily chosen, fixed prompt format." That is a model × format interaction,
reported in 2024. It is *aggregate* rather than item-level, it is not separated
from sampling noise, and it says nothing about modes or about difficulty
transfer — so the delta survives — but the honest framing is "we sharpen a
known aggregate phenomenon into an item-level structural claim with a
mechanism", not "nobody has looked at this". `paper/main.tex` has been revised
accordingly: a dedicated Related Work subsection ("Prompt sensitivity: the
nearest prior work") now states the FormatSpread cross-model finding explicitly
and articulates the three-axis delta, rather than leaving a reviewer to
discover it.

**Two entries are single-author preprints from outside academia.**
`cofailure2026` (Josef Chen, KAIKAKU) and `reasoningcodeboth2026` (Matthew
Kutakh). Both are real and correctly attributed, but neither is peer reviewed
and neither has co-authors. Both are cited only as corroborating context
alongside a stronger source, never as the sole support for a claim, and the bib
entries now say so.

## 5. Per-entry table

Verification tags: **V-ARXIV** = arXiv metadata API (`export.arxiv.org`), which
returns the canonical record for the latest version. **V-ACL** = ACL Anthology
BibTeX export (`aclanthology.org/KEY.bib`), the publisher's own file.
**V-PUB** = another publisher of record (AAAI OJS, PMLR, Nature DOI, ICLR
proceedings, ML Anthology/TMLR, HuggingFace model or dataset card).
**V-CANON** = canonical, heavily cited work with a stable and standard author
list, not re-fetched this session.

| # | Key | Status | Source used | Correction |
|---|---|---|---|---|
| 1 | `bay2026modal` | V-ARXIV | arXiv API + title block | preprint status noted |
| 2 | `blendasc2025` | V-ARXIV | arXiv API | authors added (5, not the 3 Semantic Scholar showed) |
| 3 | `certifiedsc2025` | V-ARXIV | arXiv API | authors added; given names expanded from initials |
| 4 | `samplecomplexity2025` | V-ARXIV | arXiv API | authors added (8) |
| 5 | `bestofmajority2025` | V-ARXIV | arXiv API | authors added (5) |
| 6 | `cofailure2026` | V-ARXIV | arXiv API | sole author added; preprint status noted |
| 7 | `snell2024scaling` | V-ARXIV | arXiv API + title block | none |
| 8 | `brown2024monkeys` | V-CANON | — | none |
| 9 | `wu2024inference` | V-CANON | — | none |
| 10 | `muennighoff2025s1` | V-CANON | — | none |
| 11 | `zhang2025ttssurvey` | V-PUB | testtimescaling.github.io BibTeX | none |
| 12 | `subproblemsurvey2025` | V-ARXIV | arXiv API | authors added (5) |
| 13 | `artofscaling2025` | V-ARXIV | arXiv API | authors added (3) |
| 14 | `gema2025inverse` | V-PUB + V-ARXIV | TMLR BibTeX, confirmed via arXiv API | none |
| 15 | `lightman2024verify` | V-CANON | — | none |
| 16 | `uesato2022process` | V-CANON | — | none |
| 17 | `wang2024mathshepherd` | V-ACL | `2024.acl-long.510.bib` | pages added |
| 18 | `ttsknowledge2025` | V-ARXIV | arXiv API | authors added (3) |
| 19 | `aggarwal2023adaptive` | V-CANON | — | none |
| 20 | `li2024esc` | V-PUB | iclr.cc poster page | venue confirmed ICLR 2024; caveat note removed |
| 21 | `fu2024certaindex` | V-PUB | repo BibTeX | none |
| 22 | `adaptivettc2026` | V-ARXIV | arXiv API | authors added (7) |
| 23 | `wang2023selfconsistency` | V-CANON | — | none |
| 24 | `chen2023universal` | V-CANON | — | none |
| 25 | `veccisc2026` | V-ACL | `2026.findings-acl.1305.bib` | 4th author resolved: Nianwen Xue |
| 26 | `risc2026` | V-ACL + V-ARXIV | Anthology + arXiv API | none |
| 27 | `rcs2026` | V-ARXIV | arXiv API | **title corrected**; authors added |
| 28 | `weightedreasoning2024` | V-ARXIV | arXiv API | **title corrected**; authors added |
| 29 | `prove2024` | V-ARXIV | arXiv API | authors added (3) |
| 30 | `marginalsharpening2026` | V-ARXIV | arXiv API | authors added (3) |
| 31 | `sclar2024formatspread` | V-ARXIV | arXiv API + title block | **title corrected** (comma) |
| 32 | `paraconsistency2024` | V-ACL | `2024.findings-acl.842.bib` | authors added (6) |
| 33 | `zhou2024paraphrase` | V-ACL | `2024.naacl-long.153.bib` | **given name corrected**; full list + pages added |
| 34 | `paraphraseanalysis2025` | V-ACL | `2025.ijcnlp-long.23.bib` | authors added (4); pages added |
| 35 | `verbalizedsampling2025` | V-ARXIV | arXiv API | authors added (7) |
| 36 | `deepseekr1_2025` | V-CANON | — | Nature volume/pages **not** verified; note retained |
| 37 | `shao2024deepseekmath` | V-CANON | — | none |
| 38 | `yue2025rlvr` | V-ARXIV | arXiv API journal-ref | venue upgraded to NeurIPS 2025; duplicate name confirmed genuine |
| 39 | `diversitycollapse2025` | V-ARXIV | arXiv API journal-ref | authors added; venue upgraded to COLM 2025 |
| 40 | `huang2025sharpening` | V-ARXIV | arXiv page | none |
| 41 | `ye2025limo` | V-CANON | — | none |
| 42 | `wei2022cot` | V-CANON | — | none |
| 43 | `kojima2022zeroshot` | V-CANON | — | none |
| 44 | `zhou2023leasttomost` | V-CANON | — | none |
| 45 | `yao2023tot` | V-CANON | — | none |
| 46 | `besta2024got` | V-CANON | — | none |
| 47 | `chen2023pot` | V-CANON | — | none |
| 48 | `gao2023pal` | V-CANON | — | none |
| 49 | `sprague2025tocot` | V-CANON | — | none |
| 50 | `turpin2023unfaithful` | V-CANON | — | none |
| 51 | `lanham2023measuring` | V-PUB | Semantic Scholar (30-author list) | first 10 + `others` confirmed correct |
| 52 | `lyu2023faithful` | V-CANON | — | none |
| 53 | `prematureconfidence2026` | V-ARXIV | arXiv API | authors added (7); arXiv preferred over S2 on "Chen Wu" |
| 54 | `precotprobe2026` | V-ARXIV | arXiv API | **title corrected**; authors added |
| 55 | `matcha2025` | V-ARXIV | arXiv API | authors added (5); S2's TMLR record omits Tian Qiu, noted |
| 56 | `hao2024coconut` | V-CANON | — | none |
| 57 | `pfau2024filler` | V-CANON | — | none |
| 58 | `goyal2024pause` | V-CANON | — | none |
| 59 | `overthinking2024` | V-CANON | — | none |
| 60 | `underthinking2025` | V-CANON | — | none |
| 61 | `stopoverthinking2025` | V-CANON | — | none |
| 62 | `l1_2025` | V-CANON | — | none |
| 63 | `mirzadeh2025gsmsymbolic` | V-CANON | arXiv PDF | none |
| 64 | `zhang2024gsm1k` | V-CANON | — | none |
| 65 | `wu2024reciting` | V-CANON | — | none |
| 66 | `chen2024premise` | V-CANON | — | none |
| 67 | `berglund2024reversal` | V-CANON | — | none |
| 68 | `dziri2023faithfate` | V-CANON | — | none |
| 69 | `reasoningcodeboth2026` | V-ARXIV | arXiv API | sole author added; preprint status noted |
| 70 | `huang2024cannotselfcorrect` | V-CANON | — | none |
| 71 | `kamoi2024whencan` | V-CANON | — | none |
| 72 | `madaan2023selfrefine` | V-CANON | — | none |
| 73 | `du2024debate` | V-CANON | — | none |
| 74 | `kadavath2022know` | V-CANON | — | none |
| 75 | `kuhn2023semantic` | V-CANON | — | none |
| 76 | `farquhar2024semantic` | V-PUB | Nature DOI 10.1038/s41586-024-07421-0 | vol. 630, pp. 625--630 confirmed; caveat removed; DOI added |
| 77 | `lin2022teaching` | V-CANON | — | none |
| 78 | `tian2023just` | V-CANON | — | none |
| 79 | `chen2021codex` | V-CANON | — | none |
| 80 | `cobbe2021gsm8k` | V-CANON | — | none |
| 81 | `hendrycks2021math` | V-CANON | — | none |
| 82 | `miller2024errorbars` | V-CANON | — | none |
| 83 | `hochlehnert2025sober` | V-PUB | Semantic Scholar | none |
| 84 | `dontpassk2025` | V-ARXIV | arXiv API journal-ref | authors added; venue upgraded to ICLR 2026 |
| 85 | `coverattau2025` | V-ARXIV | arXiv API | authors added (4) |
| 86 | `falsepositives2025` | V-ACL | `2025.emnlp-main.632.bib` | authors added (5); pages added |
| 87 | `kim2025correlated` | V-PUB | Semantic Scholar record of PMLR v267 | full author list added (4) |
| 88 | `atlas2025` | V-ARXIV | arXiv API | authors added (7); version-dependent numbers noted |
| 89 | `irtnet2025` | V-ARXIV | arXiv API | authors added (8) |
| 90 | `irtrouter2025` | V-ACL | `2025.acl-long.761.bib` | authors added (8); pages added |
| 91 | `psnirt2026` | V-PUB | AAAI OJS + authors' own repo BibTeX | authors added (13); volume, pages, DOI, arXiv ID added |
| 92 | `qwen25` | V-PUB | HuggingFace model card | none |
| 93 | `qwen2` | V-PUB | Qwen2.5 model card reference list | none |
| 94 | `math500` | V-PUB | HuggingFace dataset card (500 rows) | none |

## 6. Method

1. **arXiv metadata API** (`export.arxiv.org/api/query?id_list=...`) for every
   arXiv-hosted entry. This returns the canonical title, the full author list
   with unabbreviated given names, and the `journal_ref` field. It is
   preferable to both Semantic Scholar (which abbreviates given names, and
   which disagreed with arXiv on two names) and to rendered HTML (which
   sometimes drops the author block entirely — it did so for three papers here).
   The `journal_ref` field is what caught the three stale venues in entries 38,
   39 and 84.
2. **ACL Anthology BibTeX export** for every ACL/NAACL/EMNLP/IJCNLP entry. This
   is the publisher's own file and includes pages and DOI.
3. **AAAI OJS, PMLR, and the Nature DOI record** for the remaining published
   venues.
4. Entries tagged **V-CANON** are widely cited works whose author lists are
   stable and standard. They were not re-fetched. This is the one place the
   audit stops short of the arXiv record, and it is a deliberate trade: the
   marginal risk on `wei2022cot` is negligible compared with the risk on a 2026
   preprint that had no author field at all.

Where two sources disagreed, the arXiv record or the publisher of record won,
and the disagreement is recorded in the entry's `note`. Three such conflicts
arose (entries 53, 55, and the abbreviated given names in 2 and 18); none
changed which paper is being cited.

## 7. Residual risk

One item remains genuinely unverified, and it is deliberately left that way
rather than guessed:

- **`deepseekr1_2025`** is cited as the arXiv preprint, which is correct and
  complete as it stands. A version was subsequently published in *Nature*, and
  the entry's `note` says so, but the Nature volume and page numbers were not
  confirmed in this session (the lookup was rate-limited). The entry does not
  claim them. If the camera-ready should cite the journal version instead,
  those two fields must be fetched first.

Beyond that, the standing caveats are the ones in §4: three preprints
(`bay2026modal`, `cofailure2026`, `reasoningcodeboth2026`) are real and
correctly attributed but not peer reviewed, and the paper should not lean on
any of them as sole support for a claim.
