# Submitting to Quantum Machine Intelligence (Springer)

Everything below is ready to paste. Rebuild the package after any edit with

    python scripts/fill_papers.py && python scripts/make_submission.py

## Before you start

1. **Merge the work into `main` and push.** The paper links to
   https://github.com/arnavk23/quasar-security. Reviewers will see the default branch
   `main`, which is still the old version.
2. **Department** is set to Department of Electrical Engineering and Computer Science.
3. **Confirm the declarations** in the paper's Declarations section. They say no
   funding and no competing interests.
4. **ORCID iD** 0009-0007-9818-7908. Enter it in your Springer account profile and on
   the author step of the form.
5. **Check the journal page once.** Open https://link.springer.com/journal/42484 and
   read "Submission guidelines" for any rule not covered here, such as a word limit or
   the open-access fee.

## Files

All are in `papers/qmi/`.

| Upload as | File |
|---|---|
| Manuscript (LaTeX source) | `qmi_submission.zip`, or the files inside `submission/` one by one |
| Manuscript PDF, if asked | `submission/main.pdf` |
| Cover letter | `cover_letter.pdf` (text in `cover_letter.md`) |

The ZIP holds `main.tex`, `main.bbl`, `references.bib`, `sn-jnl.cls`,
`sn-mathphys-num.bst` and seven PNG figures. It compiles on its own with pdfLaTeX.
If the system asks for a "main" or "primary" file, choose `main.tex`. If it asks for
file types, the PNGs are "Figure", the `.cls`, `.bst`, `.bib` and `.bbl` are "LaTeX
supplementary" or "Other".

## Steps (Snapp, the Springer Nature submission system)

1. Go to https://link.springer.com/journal/42484 and click **Submit manuscript**. Log in
   or create an account with your IISER email. Add ORCID 0009-0007-9818-7908 to your
   profile.
2. **Manuscript file.** Click **Upload manuscript** and choose `papers/qmi/qmi_submission.zip`.
   Snapp compiles the LaTeX into a PDF itself. Do not fill any field before this upload.
   Snapp reads the title, abstract, author and statements from the file.
3. **Figures and tables.** Skip. They are inside the ZIP. Separate high-resolution files
   are only needed after acceptance. They are in `papers/qmi/submission/` (all at least
   1500 x 1200 pixels).
4. **Supplementary material.** Skip. The code and data are on GitHub.
5. **Related files.** Skip. No overlapping paper is under consideration.
6. **Check the extracted fields** against the "Form fields" section below. Fix anything
   Snapp got wrong, especially the abstract, the affiliation and the data availability
   statement.
7. Article type: **Research article** (or "Original article").
8. Paste the cover letter from `cover_letter.md` when asked.
9. Answer the declaration questions to match the paper (no funding, no competing
   interests, no ethics approval needed).
10. Open-access option: if asked, the subscription route has no article fee. Open access
    does. Check the fee before choosing.
11. Open the PDF Snapp builds. Check that all seven figures, the propositions and the
    reference list appear. Then submit.

You will get a confirmation email with a manuscript number. Keep it.

## Form fields

**Title**
The Norm Blind Spot of Amplitude-Encoding Quantum Kernels in Anomaly Detection

**Running title** (if asked)
Norm Blind Spot of Amplitude-Encoding Kernels

**Abstract** (245 words)
Amplitude encoding is a common way to load classical data into quantum kernel methods. We show that any kernel machine built on it scores an input by a Rayleigh quotient of a learned matrix. The score depends on the direction of the input and never on its norm. We ask when this bias helps anomaly detection. We test it on four security datasets from quantum key distribution telemetry and network flows, with every model trained on the same data at every sample size. In distribution the amplitude kernel never beats a Gaussian kernel, and it needs more data when samples are few. On attack types held out from training the outcome depends on the data. In both simulated settings the kernel fails, because unseen attacks differ from normal traffic mainly in norm. Appending one constant amplitude, set to the median training norm, repairs the failure. We explain why this constant works. Where the norm carries no signal, the anchor can cost accuracy. On UNSW-NB15 and CIC-IDS2017 the plain kernel does not fail. An eight-qubit ZZ feature map with tuned bandwidth is competitive with tuned classical kernels but is not the best on held-out attacks in any of six settings. Estimating every kernel entry from 1000 shots barely changes detection performance. A simple norm statistic does not predict when the failure occurs. Quantum kernels should be checked for their invariances and tested on held-out classes from real data, against classical kernels trained at the same scale.

**Keywords**
Quantum kernels; Amplitude encoding; Inductive bias; Anomaly detection; Quantum key distribution; Benchmarking

**Author**
Arnav Kapoor, Department of Electrical Engineering and Computer Science, Indian Institute of Science Education and Research Bhopal, Bhopal 462066, Madhya Pradesh, India. arnavkapoor23@iiserb.ac.in. ORCID 0009-0007-9818-7908.

**Funding**
No funding was received for this work.

**Competing interests**
The author has no competing interests to declare.

**Ethics approval / consent**
Not applicable.

**Data availability**
UNSW-NB15 (Moustafa and Slay 2015, doi:10.1109/MilCIS.2015.7348942) is distributed by UNSW Canberra at https://research.unsw.edu.au/projects/unsw-nb15-dataset. CIC-IDS2017 (Sharafaldin et al. 2018, doi:10.5220/0006639801080116) is distributed by the Canadian Institute for Cybersecurity at https://www.unb.ca/cic/datasets/ids-2017.html. The QKD telemetry and synthetic flow data were generated by simulators at https://github.com/arnavk23/quasar-security, which also holds all result files.

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
