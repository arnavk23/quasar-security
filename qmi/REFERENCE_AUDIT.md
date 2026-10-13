# Reference audit (2026-10-13)

Every reference in the two conference drafts (since removed from the repository) was
checked. Classic, widely cited works were verified from their standard
bibliographic records; anything uncertain was checked by web search. Entries
that could not be found as cited were **removed or replaced** — a fabricated
citation is grounds for desk rejection or retraction.

## Not found as cited → removed or replaced

| Key | As cited | Finding | Action |
|---|---|---|---|
| `jain2014detector` | Jain et al., "Detector-control attacks in QKD: a comprehensive study," PRA 90, 022316 (2014) | No such paper. The real Jain et al. 2014 is a Trojan-horse paper in New J. Phys. | Replaced by `jain2014trojan`: N. Jain *et al.*, "Trojan-horse attacks threaten the security of practical quantum cryptography," New J. Phys. 16, 123030 (2014) |
| `honz2019detection` | "Hardware-level detection of QKD detector blinding attacks," Opt. Express 2019 | Not found | Replaced by `dasilva2012monitoring`: T. Ferreira da Silva *et al.*, "Real-time monitoring of single-photon detectors against eavesdropping in QKD systems," arXiv:1208.0532 (2012) |
| `hari2020detection` | "Real-time detection of QKD implementation attacks via optical power monitoring," IEEE JQE 2020 | Not found | Removed |
| `shin2021channe` | Shin & Tamaki, "Channel parameter estimation for QKD security analysis," PRA 2021 | Not found | Removed |
| `liu2021qkd` | Liu & Zhang, "A deep learning approach for multi-attack classification in QKD systems," Quantum Eng. 2021 | Not found | Removed |
| `kennedy2020bayesian` | Kennedy & Jones, "Bayesian approaches to anomaly detection: a review," Stat. Sci. 2020 | Not found | Replaced by `perusquia2022bayesian`: J. A. Perusquía, J. E. Griffin, C. Villa, "Bayesian models applied to cyber security anomaly detection problems," Int. Stat. Rev. 90(1), 78–99 (2022) |
| `liu2021quantum` | Y. Liu *et al.*, "Quantum advantage of kernel methods," PRL 126, 130501 (2021) | Wrong title/venue | Replaced by Y. Liu, S. Arunachalam, K. Temme, "A rigorous and robust quantum speed-up in supervised machine learning," Nat. Phys. 17, 1013–1017 (2021) |
| `webb2023quantum` | Webb, Nadkarni, Rowe, "Quantum adversarial machine learning: a survey," Quantum Mach. Intell. 2023 | Not found. The sentence it supported ("showed amplitude encodings provide smoothing benefits") was also unsupported | Claim removed; cite S. Lu, L.-M. Duan, D.-L. Deng, "Quantum adversarial machine learning," Phys. Rev. Research 2, 033212 (2020) for the general topic |
| `vale2023quantum` | do Vale *et al.*, "Quantum support vector machines for intrusion detection," Quantum Inf. Process. 2023 | Not found | Replaced by M. Kalinin, V. Krundyshev, "Security intrusion detection using quantum machine learning techniques," J. Comput. Virol. Hack. Tech. 19(1), 125–136 (2023) |
| `ashraf2020zero` | Ashraf, Iqbal, Abbas, "Zero-trust security for IoT: a systematic literature review," IEEE Access 2020 | Not found | Replaced by N. F. Syed *et al.*, "Zero Trust Architecture (ZTA): a comprehensive survey," IEEE Access 10, 57143–57179 (2022) |
| `wang2021user` | Wang, Li, Lin, "User and entity behavior analytics for enterprise security," Comput. Secur. 2021 | Not found under those authors/venue | Replaced by M. Shashanka, M.-Y. Shen, J. Wang, same title, IEEE BigData 2016, pp. 1867–1874 |
| `wu2022transformer` | T. Wu *et al.*, "A transformer-based framework for multivariate time series representation learning," KDD 2022 | Real paper, wrong authors/year | Corrected to G. Zerveas *et al.*, KDD 2021, pp. 2114–2124 |
| `suryotrisongko2022quantum` | "Evaluating hybrid quantum ML models for malware classification," J. Inf. Secur. Appl. 67, 103193 | Wrong title/venue | Corrected to H. Suryotrisongko, Y. Musashi, "Evaluating hybrid quantum-classical deep learning for cybersecurity botnet DGA detection," Procedia Comput. Sci. 197, 223–229 (2022) |

## Real, but details corrected

| Key | Correction |
|---|---|
| `l0mti2005attacks` | Title is "Quantum key distribution with realistic states: photon-number statistics in the photon-number splitting attack" (New J. Phys. 4, 44, 2002) |
| `ren2021machine` | First author is Zi-Ang Ren (IEEE Commun. Lett. 25(3), 940–944, 2021). It is about protocol/parameter selection, not attack detection; the sentence citing it was reworded |
| `gilman2019zero` | Published 2017, not 2019 |
| `nist800207` / `rose2020zero` | Same document cited twice; merged |
| `larger2009controlling` / `makarov2006detector` | Keys renamed to match the actual papers (Lydersen *et al.* 2010; Makarov, Anisimov, Skaar 2006) |
| `kapoor2026hyqad` | Self-citation to an unpublished manuscript; removed from the merged paper, which contains both studies |

## Verified as cited (no change)

Bennett & Brassard 1984; Gisin *et al.* RMP 2002; Lo, Ma & Chen PRL 2005;
Scarani *et al.* RMP 2009; Huttner *et al.* PRA 1995; Makarov, Anisimov & Skaar
PRA 2006; Lydersen *et al.* Nat. Photon. 2010; Gisin *et al.* PRA 2006 (Trojan
horse); Bennett *et al.* J. Cryptol. 1992; Mao *et al.* New J. Phys. 2020;
Havlíček *et al.* Nature 2019; Schuld & Petruccione 2021; Biamonte *et al.*
Nature 2017; Bergholm *et al.* arXiv:1811.04968; Huang *et al.* Nat. Commun.
2021; Rasmussen & Williams 2006; Rousseeuw & Van Driessen 1999; Wolpert 1992;
Rahimi & Recht 2007; Breiman 2001; Guo *et al.* ICML 2017; Zadrozny & Elkan KDD
2002; Lo, Curty & Qi PRL 2012; Liao *et al.* Nature 2017; Lucamarini *et al.*
Nature 2018; Preskill Quantum 2018; Rose *et al.* NIST SP 800-207; Kindervag
(Forrester, Nov. 2010); Roesch LISA 1999; Sommer & Paxson S&P 2010; Mirsky
*et al.* NDSS 2018; Goodfellow *et al.* ICLR 2015; Zhou *et al.* AI Open 2020;
Cohen *et al.* ICML 2019; Cristianini *et al.* NeurIPS 2001; Williams & Seeger
NeurIPS 2000; Platt 1999; Schölkopf & Smola 2002; Bishop 2006; Chen & Guestrin
KDD 2016; Buczak & Guven 2016; Sharafaldin *et al.* ICISSP 2018; Moustafa &
Slay MilCIS 2015.

## DOIs

All 46 DOIs in `qmi/references.bib` were checked against Crossref on
2026-10-13. Each resolves, and its registered title matches the cited title
(the XGBoost record stores the title without its subtitle). The 12 entries
without a DOI have none registered: the NeurIPS, ICML and ICLR proceedings
(Cristianini et al., Williams & Seeger, Rahimi & Recht, Guo et al., Cohen et
al.), Platt's 1999 chapter, Bishop (2006), Gilman & Barth (2017), Roesch (USENIX
LISA 1999), the Forrester report, and the two companion manuscripts.
Bennett & Brassard (1984) cites its 2014 reprint in *Theor. Comput. Sci.*.
Ferreira da Silva et al. is now cited from *Opt. Express* 20(17), 18911 (2012),
not arXiv.

## Added for the Quantum Machine Intelligence version (2026-10-14)

Twelve references were added. Each was checked against Crossref or the arXiv API
for title, authors, venue, volume, pages and year.

- Crossref, DOI verified: Thanasilp et al., *Nat. Commun.* 15, 5200 (2024);
  Shaydulin & Wild, *Phys. Rev. A* 106, 042407 (2022); LaRose & Coyle, *Phys. Rev. A*
  102, 032420 (2020); Schuld, Sweke & Meyer, *Phys. Rev. A* 103, 032430 (2021);
  Möttönen et al., *Quantum Inf. Comput.* 5(6), 467 (2005); Wang et al., *Quantum* 5,
  531 (2021); Hubregtsen et al., *Phys. Rev. A* 106, 042431 (2022); Tang, STOC 2019,
  217-228.
- arXiv, no DOI: Schuld, arXiv:2101.11020; Bowles, Ahmed & Schuld, arXiv:2403.07059;
  Canatar et al., arXiv:2206.06686.
- NeurIPS, no DOI: Kübler, Buchholz & Schölkopf, NeurIPS 34 (2021), confirmed via the
  arXiv journal reference of 2106.03747.
