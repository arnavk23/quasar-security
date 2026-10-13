Dear Editors,

Please consider the enclosed manuscript, "The Norm Blind Spot of Amplitude-Encoding
Quantum Kernels in Anomaly Detection", for publication in Quantum Machine Intelligence.

Amplitude encoding is widely used to load classical data into quantum kernels. We show
that every kernel machine built on it computes a Rayleigh quotient of a learned matrix.
Such models see the direction of an input and never its norm. We test what this means
for anomaly detection on four security datasets, with every model trained on the same
data at every sample size. The bias causes a clear failure on unseen attack types in
simulated data, and a one-amplitude norm anchor repairs it. On two public benchmarks the
failure does not appear. We also compare an eight-qubit ZZ feature map with tuned
bandwidth and simulate finite-shot kernel estimation.

We think the paper suits the journal because it gives an exact account of an encoding's
inductive bias and tests it carefully against classical kernels. That kind of
benchmarking is increasingly asked for in quantum machine learning.

Related work by the author. A short conference paper on the network-flow part of this
study [TITLE, VENUE, STATUS] has been submitted. The present manuscript extends it with
the QKD domain, the function-class analysis (Propositions 2 and 3), the explanation of
the anchor constant, the entangling feature-map and finite-shot experiments, and the
norm diagnostic. Overlapping text and tables are limited to [SECTIONS]. A copy can be
provided to the editor on request.

The manuscript is not under consideration elsewhere. Code and data preparation scripts
are public at https://github.com/arnavk23/quasar-security.

Sincerely,
Arnav Kapoor
Indian Institute of Science Education and Research Bhopal
arnavkapoor23@iiserb.ac.in
