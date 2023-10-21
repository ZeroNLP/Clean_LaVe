# 说明
## STEP 1
annotate test/dev/train set using LaVeEntail(TE)
```
sh annotate.sh
```

## STEP 2
Clean Data Detection Module
```
sh NL.sh
```

## STEP 3
finetune TE and infer
```
sh finetuneAtest_search.sh
```
