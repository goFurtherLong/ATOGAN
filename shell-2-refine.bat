@REM the path of word embbedings
set emb_path_prefix=D:\\coding\\Xiong\\MUSE\\data

@REM   hu
set other_language=da


set src_language=en
set tgt_language=%other_language%

@REM there to reload the trained mapping function mapping.pth(x to y) and mapping2.pth(y to x)
set reload_path_prefix=.\dumped\muse-adv\%src_language%_%tgt_language%


@REM  source to target
set exp_id_prefix=%src_language%_%tgt_language%
for /l %%i in (1,1,5) do python refine.py  --exp_name muse-refine --dico_max_rank 20000    --dico_build "S2T&T2S"  --normalize_embeddings center  --reload_path %reload_path_prefix%_time%%i\\best_mapping.pth   --exp_id %exp_id_prefix%_time%%i --src_lang %src_language% --tgt_lang %tgt_language%   --src_emb %emb_path_prefix%\\wiki.%src_language%.vec --tgt_emb %emb_path_prefix%\\wiki.%tgt_language%.vec
@REM  target to source
set exp_id_prefix=%tgt_language%_%src_language%
for /l %%i in (1,1,5) do python refine.py  --exp_name muse-refine --dico_max_rank 20000    --dico_build "S2T&T2S"  --normalize_embeddings center  --reload_path %reload_path_prefix%_time%%i\\best_mapping2.pth   --exp_id %exp_id_prefix%_time%%i --src_lang %tgt_language% --tgt_lang %src_language%   --src_emb %emb_path_prefix%\\wiki.%tgt_language%.vec --tgt_emb %emb_path_prefix%\\wiki.%src_language%.vec







@REM @REM @REM For vecmap dataset below

@REM @REM the path of word embbedings 
@REM set emb_path_prefix=D:\\coding\\Xiong\\VECMAP\\data

@REM @REM ru  hu
@REM set other_language=da

@REM set src_language=en
@REM set tgt_language=%other_language%
@REM @REM there to reload the trained mapping function mapping.pth(x to y) and mapping2.pth(y to x)
@REM set reload_path_prefix=.\dumped\vecmap-adv\%src_language%_%tgt_language%

@REM @REM  source to target
@REM set exp_id_prefix=%src_language%_%tgt_language%
@REM for /l %%i in (1,1,5) do python refine.py  --exp_name vecmap-refine --dico_max_rank 20000    --dico_build "S2T&T2S"  --normalize_embeddings "renorm,center,renorm"  --reload_path %reload_path_prefix%_time%%i\\best_mapping.pth   --exp_id %exp_id_prefix%_time%%i --src_lang %src_language% --tgt_lang %tgt_language%   --src_emb %emb_path_prefix%\\%src_language%.emb.txt --tgt_emb %emb_path_prefix%\\%tgt_language%.emb.txt
@REM @REM  target to source
@REM set exp_id_prefix=%tgt_language%_%src_language%
@REM for /l %%i in (1,1,5) do python refine.py  --exp_name vecmap-refine --dico_max_rank 20000    --dico_build "S2T&T2S"  --normalize_embeddings "renorm,center,renorm"  --reload_path %reload_path_prefix%_time%%i\\best_mapping2.pth   --exp_id %exp_id_prefix%_time%%i --src_lang %tgt_language% --tgt_lang %src_language%   --src_emb %emb_path_prefix%\\%tgt_language%.emb.txt --tgt_emb %emb_path_prefix%\\%src_language%.emb.txt
