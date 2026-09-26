### US providers (gemini-1.5-flash, llama-3.1-8b, gpt-4o-mini, llama-3.1-70b, gpt-4o)

| Strategy | Accuracy | $ per 1,000 requests | Saving vs gpt-4o |
|---|---|---|---|
| everything to gpt-4o | 77.6% | 3.638 | - |
| learned router (confidence 0.82) | 77.8% | 0.827 | 77.3% |
| cascade: gemini-1.5-flash + llama-3.1-70b, escalate when they disagree | 77.6% | 2.332 | 35.9% |
| fixed task-to-model map | 78.5% | 2.478 | 31.9% |
| oracle (cheapest model that was right) | 88.7% | 0.436 | 88.0% |
| everything to gemini-1.5-flash | 63.8% | 0.098 | 97.3% |
| everything to llama-3.1-8b | 50.0% | 0.233 | 93.6% |
| everything to gpt-4o-mini | 71.4% | 0.226 | 93.8% |
| everything to llama-3.1-70b | 74.0% | 1.129 | 69.0% |

Router against gpt-4o on the same questions: +0.3 points (95% interval -1.0 to +1.6); traffic share gemini-1.5-flash 37.0%, llama-3.1-8b 7.0%, gpt-4o-mini 17.3%, llama-3.1-70b 13.7%, gpt-4o 25.0%.

### any provider (gemini-1.5-flash, llama-3.1-8b, gpt-4o-mini, deepseek-v3, llama-3.1-70b, gpt-4o)

| Strategy | Accuracy | $ per 1,000 requests | Saving vs gpt-4o |
|---|---|---|---|
| everything to gpt-4o | 77.6% | 3.638 | - |
| learned router (confidence 0.78) | 78.3% | 0.382 | 89.5% |
| cascade: gpt-4o-mini + deepseek-v3, escalate when they disagree | 77.6% | 1.102 | 69.7% |
| fixed task-to-model map | 79.1% | 2.193 | 39.7% |
| oracle (cheapest model that was right) | 89.9% | 0.379 | 89.6% |
| everything to gemini-1.5-flash | 63.8% | 0.098 | 97.3% |
| everything to llama-3.1-8b | 50.0% | 0.233 | 93.6% |
| everything to gpt-4o-mini | 71.4% | 0.226 | 93.8% |
| everything to deepseek-v3 | 78.6% | 0.330 | 90.9% |
| everything to llama-3.1-70b | 74.0% | 1.129 | 69.0% |

Router against gpt-4o on the same questions: +0.7 points (95% interval -0.7 to +2.1); traffic share gemini-1.5-flash 37.4%, llama-3.1-8b 12.5%, gpt-4o-mini 13.4%, deepseek-v3 18.5%, llama-3.1-70b 2.7%, gpt-4o 15.5%.

