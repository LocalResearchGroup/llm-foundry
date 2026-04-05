| dataset | split | total_tokens | num_sequences | avg_length | median_length | std_length | min_length | max_length | seqs_over_max | pct_over_max | p25 | p50 | p75 | p95 | p99 | eos_count | im_start_count | im_end_count |
|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|
| tulu | train | 174084181 | 714897 | 243.51 | 143 | 420.63 | 26 | 8191 | 0 | 0.0 | 74 | 143 | 292 | 579 | 2056 | 1429798 | 1429798 | 1429798 |
| tulu | test | 19176639 | 79428 | 241.43 | 143 | 413.49 | 26 | 8191 | 0 | 0.0 | 74 | 143 | 292 | 573 | 1970 | 158856 | 158856 | 158856 |
| numina | train | 89131068 | 773540 | 115.22 | 100 | 65.2 | 29 | 3689 | 0 | 0.0 | 75 | 100 | 139 | 221 | 326 | 1547080 | 1547080 | 1547080 |
| numina | test | 9885623 | 85950 | 115.02 | 100 | 64.31 | 33 | 1887 | 0 | 0.0 | 75 | 100 | 138 | 220 | 329 | 171900 | 171900 | 171900 |
| glaive | train | 141275887 | 855334 | 165.17 | 140 | 135.23 | 26 | 8086 | 0 | 0.0 | 101 | 140 | 196 | 340 | 617 | 1710668 | 1710668 | 1710668 |
| glaive | test | 15721960 | 95039 | 165.43 | 140 | 136.77 | 33 | 5761 | 0 | 0.0 | 101 | 140 | 196 | 338 | 620 | 190078 | 190078 | 190078 |
| finemath | train | 9824020868 | 6029479 | 1629.33 | 920 | 4436.31 | 35 | 1035656 | 126349 | 2.1 | 563 | 920 | 1653 | 4856 | 11664 | 524 | 498 | 524 |
| finemath | test | 1084253271 | 669938 | 1618.44 | 920 | 3951.22 | 39 | 709691 | 14019 | 2.09 | 563 | 920 | 1654 | 4841 | 11736 | 0 | 0 | 0 |
| pythonedu | train | 3637262212 | 6910582 | 526.33 | 274 | 2971.62 | 7 | 1720895 | 12059 | 0.17 | 144 | 274 | 542 | 1665 | 3818 | 0 | 0 | 0 |
| pythonedu | test | 401734696 | 767838 | 523.2 | 274 | 2743.18 | 15 | 982481 | 1293 | 0.17 | 144 | 274 | 541 | 1656 | 3790 | 0 | 0 | 0 |

## Special Token Check

**tulu/train:**
- EOS tokens: 1,429,798 (2.00 per seq)
- <|im_start|>: 1,429,798 (2.00 per seq)
- <|im_end|>: 1,429,798 (2.00 per seq)

**tulu/test:**
- EOS tokens: 158,856 (2.00 per seq)
- <|im_start|>: 158,856 (2.00 per seq)
- <|im_end|>: 158,856 (2.00 per seq)

**numina/train:**
- EOS tokens: 1,547,080 (2.00 per seq)
- <|im_start|>: 1,547,080 (2.00 per seq)
- <|im_end|>: 1,547,080 (2.00 per seq)

**numina/test:**
- EOS tokens: 171,900 (2.00 per seq)
- <|im_start|>: 171,900 (2.00 per seq)
- <|im_end|>: 171,900 (2.00 per seq)

**glaive/train:**
- EOS tokens: 1,710,668 (2.00 per seq)
- <|im_start|>: 1,710,668 (2.00 per seq)
- <|im_end|>: 1,710,668 (2.00 per seq)

**glaive/test:**
- EOS tokens: 190,078 (2.00 per seq)
- <|im_start|>: 190,078 (2.00 per seq)
- <|im_end|>: 190,078 (2.00 per seq)

**finemath/train:**
- EOS tokens: 524 (0.00 per seq)
- <|im_start|>: 498 (0.00 per seq)
- <|im_end|>: 524 (0.00 per seq)

**finemath/test:**
- EOS tokens: 0 (0.00 per seq)
- <|im_start|>: 0 (0.00 per seq)
- <|im_end|>: 0 (0.00 per seq)

**pythonedu/train:**
- EOS tokens: 0 (0.00 per seq)
- <|im_start|>: 0 (0.00 per seq)
- <|im_end|>: 0 (0.00 per seq)

**pythonedu/test:**
- EOS tokens: 0 (0.00 per seq)
- <|im_start|>: 0 (0.00 per seq)
- <|im_end|>: 0 (0.00 per seq)
