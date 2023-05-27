@REM the path of word embbedings (muse dataset or vecmap dataset)
set emb_path_prefix=D:\\coding\\Xiong\\MUSE\\data


@REM   hu
set other_language=da

@REM since our model is bi-directional, we can get both s2t and t2s mappings.
set src_language=en
set tgt_language=%other_language%
@REM name of the exp_id
set exp_id_prefix=%src_language%_%tgt_language%


for /l %%i in (1,1,5) do python adv-training.py   --exp_name muse-adv   --exp_id %exp_id_prefix%_time%%i  --src_lang  %src_language%  --tgt_lang %tgt_language% --src_emb %emb_path_prefix%\\wiki.%src_language%.vec --tgt_emb %emb_path_prefix%\\wiki.%tgt_language%.vec



@REM @REM For vecmap dataset below

@REM @REM the path of word embbedings (muse dataset or vecmap dataset)
@REM set emb_path_prefix=D:\\coding\\Xiong\\VECMAP\\data

@REM @REM ru  hu
@REM set other_language=da
@REM @REM determine the language pairs
@REM set src_language=en
@REM set tgt_language=%other_language%

@REM @REM name of the exp_id
@REM set exp_id_prefix=%src_language%_%tgt_language%

@REM for /l %%i in (1,1,5) do python adv-training.py   --exp_name vecmap-adv  --epoch_size 500000 --n_epochs 10   --exp_id %exp_id_prefix%_time%%i --normalize_embeddings "renorm,center,renorm"  --src_lang  %src_language%  --tgt_lang %tgt_language% --src_emb %emb_path_prefix%\\%src_language%.emb.txt --tgt_emb %emb_path_prefix%\\%tgt_language%.emb.txt

