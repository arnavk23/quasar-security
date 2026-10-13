# Submitting to Quantum Machine Intelligence (Springer)

Everything below is ready to paste. Rebuild the package after any edit with

    python scripts/fill_papers.py && python scripts/make_submission.py

## Before you start

1. **Merge the work into `main` and push.** The paper links to
   https://github.com/arnavk23/quasar-security. Reviewers will see the default branch
   `main`, which is still the old version.
2. **Confirm your department name.** The paper says "Department of Computer Science".
   Your older drafts disagreed. Use the exact name IISER Bhopal uses.
3. **Confirm the declarations** in the paper's Declarations section. They say no
   funding and no competing interests.
4. **Get an ORCID iD** (free, https://orcid.org) if you do not have one. Most Springer
   systems ask for it.
5. **Check the journal page once.** Open https://link.springer.com/journal/42484 and
   read "Submission guidelines" for any rule not covered here, such as a word limit or
   the open-access fee.

## Files to upload

All are in `papers/qmi/`.

| Upload as | File |
|---|---|
| Manuscript (LaTeX source) | `qmi_submission.zip`, or the files inside `submission/` one by one |
| Manuscript PDF, if asked | `submission/main.pdf` |
| Cover letter | paste `cover_letter.md` into the text box, or upload it as a file |

The ZIP holds `main.tex`, `main.bbl`, `references.bib`, `sn-jnl.cls`,
`sn-mathphys-num.bst` and seven PNG figures. It compiles on its own with pdfLaTeX.
If the system asks for a "main" or "primary" file, choose `main.tex`. If it asks for
file types, the PNGs are "Figure", the `.cls`, `.bst`, `.bib` and `.bbl` are "LaTeX
supplementary" or "Other".

## Steps

1. Go to https://link.springer.com/journal/42484 and click **Submit manuscript**.
2. Log in or create an account with your IISER email.
3. Article type: **Research article** (or "Original article").
4. Paste the title, abstract and keywords below.
5. Author: add yourself as the only and corresponding author, with your ORCID.
6. Upload the files from the table above.
7. Answer the declaration questions to match the paper.
8. Suggested reviewers are optional. See the note below.
9. Open-access option: if the journal offers a choice, the subscription route has no
   article fee. Choosing open access does. Check the fee on the journal page before
   choosing.
10. Approve the PDF that the system builds. Check that all seven figures, both
    propositions and the reference list appear. Then submit.

You will get a confirmation email with a manuscript number. Keep it.

## Form fields

**Title**
The Norm Blind Spot of Amplitude-Encoding Quantum Kernels in Anomaly Detection

**Running title** (if asked)
Norm Blind Spot of Amplitude-Encoding Kernels

**Abstract** (249 words)
Amplitude encoding is a common way to load classical data into quantum kernel methods. We show that any kernel machine built on it scores an input by a Rayleigh quotient of a learned matrix. The score depends on the direction of the input and never on its norm. We ask when this bias helps anomaly detection. We test it on four security datasets from QKD telemetry and network flows, with every model trained on the same data at every sample size. In distribution the amplitude kernel never beats a Gaussian kernel, and it needs more data when samples are few. On attack types held out from training the outcome depends on the data. In both simulated settings the kernel fails, because unseen attacks differ from normal traffic mainly in norm. Appending one constant amplitude, set to the median training norm, repairs the failure. We explain why this constant works. Where the norm carries no signal, the anchor can cost accuracy. On UNSW-NB15 and CIC-IDS2017 the plain kernel does not fail. An eight-qubit ZZ feature map with tuned bandwidth is competitive with tuned classical kernels but is not the best on held-out attacks in any of six settings. Estimating every kernel entry from 1000 shots lowers mean AUC by at most 0.015 in any setting. A simple norm statistic does not predict when the failure occurs. Quantum kernels should be checked for their invariances and tested on held-out classes from real data, against classical kernels trained at the same scale.

**Keywords**
Quantum kernels; Amplitude encoding; Inductive bias; Anomaly detection; Quantum key distribution; Benchmarking

**Funding**
No funding was received for this work.

**Competing interests**
The author has no competing interests to declare.

**Ethics approval / consent**
Not applicable.

**Data availability**
UNSW-NB15 and CIC-IDS2017 are publicly available from their original providers. The simulators that generate the QKD and synthetic flow data, and all result files, are at https://github.com/arnavk23/quasar-security.

**Code availability**
https://github.com/arnavk23/quasar-security

**Author contributions**
A.K. designed the study, wrote the code, ran the experiments and wrote the manuscript.

**Previously published or under review elsewhere?**
No.

## Suggested reviewers (optional)

Pick researchers who work on quantum kernels and have no link to you. Good places to
look are the authors of the quantum-kernel papers the manuscript cites, for example
the work on kernel concentration, bandwidth, inductive bias and benchmarking. Take
their current institutional email from their own recent papers or group pages. Do not
suggest anyone you have worked with.

## After submitting

- First decision usually takes weeks to a few months. You can track status in the
  submission system.
- If reviews ask for changes, edit `qmi.template.tex`, rerun the two commands at the
  top, and upload the new package with a point-by-point response letter.
