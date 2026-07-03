# Bayesian Probability Results — Part 2

| Keyword | Category | Prior P(Positive) | Likelihood P(kw\|Positive) | Marginal P(kw) | Posterior P(Positive\|kw) |
|---------|----------|--------------------|------------------------------|-----------------|-----------------------------|
| masterpiece | POSITIVE | 0.5000 | 0.000214 | 0.000308 | 0.347 |
| hilarious | POSITIVE | 0.5000 | 0.000446 | 0.000522 | 0.427 |
| brilliant | POSITIVE | 0.5000 | 0.000628 | 0.000712 | 0.441 |
| boring | NEGATIVE | 0.5000 | 0.000325 | 0.000540 | 0.301 |
| disappointing | NEGATIVE | 0.5000 | 0.000197 | 0.000368 | 0.268 |
| waste | NEGATIVE | 0.5000 | 0.000253 | 0.000416 | 0.304 |
| funny | CONTROL | 0.5000 | 0.000357 | 0.000394 | 0.453 |

**Reading the table:**
- **Prior** `P(Positive)` — baseline probability any review is positive (50% by dataset construction).
- **Likelihood** `P(keyword\|Positive)` — fraction of all words in positive reviews that are this keyword.
- **Marginal** `P(keyword)` — average number of times this keyword appears per review, across the whole dataset.
- **Posterior** `P(Positive\|keyword)` — updated probability a review is positive, given that the keyword appeared. This is the only conditional probability the assignment asks for; `P(Negative\|keyword)` is intentionally not computed.
