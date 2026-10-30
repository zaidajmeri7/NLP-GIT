# NLP Lab Assignment 8 - Solutions

**Course:** Natural Language Processing (AI357)  
**Department:** Artificial Intelligence, SVNIT Surat  
**Platform / Language:** Python 3 (using standard libraries: pandas, collections, math - no complex/third-party NLP libraries)

---

## Folder Structure
All question solutions are organized into dedicated Jupyter Notebooks with all outputs pre-computed:

```text
NLP_Assignment_8_Solution/
│
├── midsem/                                # Midsem exam problem notebooks
│   ├── Q1_Finite_State_Transducer.ipynb  # Q1: English Noun Plural FST (+SG, +PL, +N)
│   ├── Q2_WordPiece.ipynb                # Q2: WordPiece tokenization (2 merges)
│   ├── Q3_TF_IDF.ipynb                   # Q3: TF-IDF & sentence similarity
│   ├── Q4_Naive_Bayes.ipynb              # Q4: Naive Bayes spam classification (E8 prediction)
│   └── Q5_Kneser_Ney.ipynb               # Q5: Kneser-Ney bigram smoothing
│
├── Spell-Correction/                      # Dataset files for spell correction
│   ├── count_1edit.txt                   # Character confusion matrix counts (1-edit)
│   ├── count_unigram.txt                 # English word unigram frequencies
│   └── spell-errors.txt                  # Known error-to-correct word pairs
│
├── Spell-Correction_Model.ipynb          # Noisy Channel Spell Correction Model
│
├── Q1_Finite_State_Transducer.ipynb      # Root-level copy for convenience
├── Q2_WordPiece.ipynb                    # Root-level copy for convenience
├── Q3_TF_IDF.ipynb                       # Root-level copy for convenience
├── Q4_Naive_Bayes.ipynb                  # Root-level copy for convenience
├── Q5_Kneser_Ney.ipynb                   # Root-level copy for convenience
└── README.md                             # This documentation file
```

---

## Summary of Questions & Results

### Part 1: Mid-Semester Exam Problems

#### Question 1: Finite State Transducer for Noun Pluralization (`Q1_Finite_State_Transducer.ipynb`)
- **Task:** Build an FST for English noun pluralization with regular rule (`+s`) and exception rule (`+es` for stems ending in `s`, `x`, `z`, `ch`, `sh`).
- **Features:** `+N` (noun), `+SG` (singular), `+PL` (plural).
- **Contents:**
  - Formal definition of FST states ($q_0, q_{reg}, q_s, q_x, q_z, q_c, q_{ch}, q_{sh}, q_N, q_{acc\_sg}, q_{acc\_pl}$).
  - Complete transition table in a pandas DataFrame.
  - Python implementation supporting morphological generation (`stem + feature -> surface word`) and morphological analysis (`surface word -> stem + features`).
  - Tested on regular nouns (`cat`, `dog`, `book`) and exception nouns (`bus`, `box`, `buzz`, `church`, `dish`).

#### Question 2: WordPiece Tokenization (`Q2_WordPiece.ipynb`)
- **Corpus:** 16 word tokens (`low`: 5, `lower`: 2, `lowest`: 3, `new`: 4, `newer`: 2).
- **WordPiece Score Formula:** $\text{score}(u, v) = \frac{\text{count}(u, v)}{\text{count}(u) \times \text{count}(v)}$
- **Merge 1 [2 Marks]:**
  - Pair: `('##s', '##t')` with score $\frac{3}{3 \times 3} = \mathbf{0.333333} \to \text{'##st'}$.
- **Merge 2 [2 Marks]:**
  - Pair: `('l', '##o')` with score $\frac{10}{10 \times 10} = \mathbf{0.100000} \to \text{'lo'}$.
- **Tokenization of 'newest' [2 Marks]:**
  - Initial character pieces: `['n', '##e', '##w', '##e', '##s', '##t']`
  - After Merge 1: `['n', '##e', '##w', '##e', '##st']`
  - After Merge 2: **`['n', '##e', '##w', '##e', '##st']`**

#### Question 3: TF-IDF & Sentence Similarity (`Q3_TF_IDF.ipynb`)
- **Dataset:** 4 sentences: $S_1$ = `cat dog animal`, $S_2$ = `cat cat pet`, $S_3$ = `dog pet animal`, $S_4$ = `cat dog pet`.
- **Formulas:** Raw TF count, Unsmoothed IDF: $\text{IDF}(w) = \ln(N / \text{DF}(w))$.
- **IDF Weights:**
  - `animal`: $\ln(4/2) = \ln(2) \approx \mathbf{0.693147}$
  - `cat`, `dog`, `pet`: $\ln(4/3) \approx \mathbf{0.287682}$
- **Pairwise Cosine Similarities:**
  - $\text{Sim}(S_1, S_2) \approx 0.320147$
  - $\text{Sim}(S_1, S_3) \approx \mathbf{0.871882}$ (Highest!)
  - $\text{Sim}(S_1, S_4) \approx 0.413308$
  - $\text{Sim}(S_2, S_3) \approx 0.160074$
  - $\text{Sim}(S_2, S_4) \approx 0.774597$
  - $\text{Sim}(S_3, S_4) \approx 0.413308$
- **Closest Pair [2 Marks]:** **Sentence $S_1$ (`cat dog animal`) and Sentence $S_3$ (`dog pet animal`)** with similarity **`0.871882`**.

#### Question 4: Naive Bayes Spam Classification (`Q4_Naive_Bayes.ipynb`)
- **Features:** Binary flags for `free`, `win`, `offer`.
- **Priors:** $P(\text{Spam}) = 4/7 \approx 0.5714$, $P(\text{NotSpam}) = 3/7 \approx 0.4286$.
- **Likelihoods for feature = 0:**
  - $P(\text{free}=0 \mid \text{Spam}) = 1/4$, $P(\text{win}=0 \mid \text{Spam}) = 1/4$, $P(\text{offer}=0 \mid \text{Spam}) = 1/4$
  - $P(\text{free}=0 \mid \text{NotSpam}) = 2/3$, $P(\text{win}=0 \mid \text{NotSpam}) = 2/3$, $P(\text{offer}=0 \mid \text{NotSpam}) = 2/3$
- **Test Email $E_8 = (0, 0, 0)$:**
  - $\text{Score}(\text{Spam}) = \frac{4}{7} \times \left(\frac{1}{4}\right)^3 = \frac{1}{112} \approx \mathbf{0.008929}$
  - $\text{Score}(\text{NotSpam}) = \frac{3}{7} \times \left(\frac{2}{3}\right)^3 = \frac{8}{63} \approx \mathbf{0.126984}$
- **Prediction [2 Marks]:** **`NotSpam`** (Normalized probability: $93.43\%$ NotSpam vs $6.57\%$ Spam).

#### Question 5: Kneser-Ney Language Model Smoothing (`Q5_Kneser_Ney.ipynb`)
- **Corpus:** 5 sentences, no start/end markers, discount $d = 0.5$.
- **Distinct Bigram Types:** 8 bigrams.
- **Continuation Probabilities:**
  - $P_{\text{cont}}(\text{AI}) = 2/8 = 0.25$ (preceded by `like`, `love`)
  - $P_{\text{cont}}(\text{NLP}) = 2/8 = 0.25$ (preceded by `like`, `love`)
- **Computations:**
  - **$P_{KN}(\text{AI} \mid \text{love})$ [3 Marks]:**
    - Discounted MLE term: $\frac{\max(1 - 0.5, 0)}{2} = 0.25$
    - Lambda weight: $\lambda(\text{love}) = \frac{0.5 \times 2}{2} = 0.5$
    - Backoff term: $0.5 \times 0.25 = 0.125$
    - **Total:** $0.25 + 0.125 = \frac{3}{8} = \mathbf{0.375}$
  - **$P_{KN}(\text{NLP} \mid \text{like})$ [3 Marks]:**
    - Discounted MLE term: $\frac{\max(2 - 0.5, 0)}{3} = 0.5$
    - Lambda weight: $\lambda(\text{like}) = \frac{0.5 \times 2}{3} = 1/3 \approx 0.333333$
    - Backoff term: $\frac{1}{3} \times 0.25 = \frac{1}{12} \approx 0.083333$
    - **Total:** $0.5 + \frac{1}{12} = \frac{7}{12} \approx \mathbf{0.583333}$

---

### Part 2: Spell Correction Model (`Spell-Correction_Model.ipynb`)
- **Model:** Noisy Channel Model $\hat{w} = \arg\max_{w \in \text{Candidates}} P(x \mid w) \times P(w)$
- **Components:**
  - **Prior $P(w)$:** Word frequencies from `count_unigram.txt` (333,333 words).
  - **Channel Likelihood $P(x \mid w)$:** Error confusion frequencies from `count_1edit.txt` (1,542 edit rules) for substitutions, deletions, insertions, and transpositions.
  - **Candidate Generation:** Edit distance 1 candidates from unigram vocabulary + dataset candidates.
- **Evaluation:** Evaluated on all 39,710 error pairs from `spell-errors.txt`.
- **Accuracy:** **`72.48%`** overall test accuracy.
- **Sample predictions:**
  - `rainning` $\to$ `raining`
  - `writtings` $\to$ `writings`
  - `disparingly` $\to$ `disparagingly`
  - `yello` $\to$ `yellow`
  - `elegable` $\to$ `eligible`
  - `recieve` $\to$ `receive`
