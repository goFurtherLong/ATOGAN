@REM the path of word embbedings (muse dataset or vecmap dataset)
set emb_path_prefix=D:\\coding\\Xiong\\FMGAN\\data

@REM  hu
set other_language=da


@REM source to target

set src_language=en
set tgt_language=%other_language%
set reload_path_prefix=C:\Users\12425\Desktop\ATOGAN-github\dumped\muse-refine\%src_language%_%tgt_language%

@REM for muse dataset, we recommand the dico_max_rank set to 20~40k, dico_build to "S2T|T2S", the normalize setting:  center,renorm
set exp_id_prefix=%src_language%_%tgt_language%
for /l %%i in (1,1,1) do python local_mapping.py  --exp_name muse-localMapping --dico_max_rank 30000  --step_size 0.1  --dico_build "S2T|T2S"  --normalize_embeddings "center,renorm"  --reload_path %reload_path_prefix%_time%%i\\best_mapping.pth  --k_closest 70  --exp_id %exp_id_prefix% --src_lang %src_language% --tgt_lang %tgt_language%   --src_emb %emb_path_prefix%\\wiki.%src_language%.vec --tgt_emb %emb_path_prefix%\\wiki.%tgt_language%.vec

@REM for vecmap dataset, we recommand the dico_max_rank set to 75k, dico_build to "S2T&T2S", the normalize setting is same to adv training:  renorm,center,renorm 
@REM for /l %%i in (1,1,5) do python local_mapping.py  --exp_name vecmap-localMapping --dico_max_rank 75000    --normalize_embeddings "renorm,center,renorm" --step_size 0.2 --iter 2  --reload_path %reload_path%_time%%i\\best_mapping.pth  --k_closest 100  --exp_id %exp_id_prefix% --src_lang %src_language% --tgt_lang %tgt_language%   --src_emb %emb_path_prefix%\\%src_language%.emb.txt --tgt_emb %emb_path_prefix%\\%tgt_language%.emb.txt


@REM target to source

set src_language=%other_language%
set tgt_language=en
set reload_path_prefix=C:\Users\12425\Desktop\ATOGAN-github\dumped\muse-refine\%src_language%_%tgt_language%

set exp_id_prefix=%src_language%_%tgt_language%
for /l %%i in (1,1,5) do python local_mapping.py  --exp_name muse-localMapping --dico_max_rank 30000  --step_size 0.1  --dico_build "S2T|T2S"  --normalize_embeddings "center,renorm"  --reload_path %reload_path_prefix%_time%%i\\best_mapping.pth  --k_closest 70  --exp_id %exp_id_prefix% --src_lang %src_language% --tgt_lang %tgt_language%   --src_emb %emb_path_prefix%\\wiki.%src_language%.vec --tgt_emb %emb_path_prefix%\\wiki.%tgt_language%.vec
@REM for /l %%i in (1,1,5) do python local_mapping.py  --exp_name vecmap-localMapping --dico_max_rank 75000    --normalize_embeddings "renorm,center,renorm" --step_size 0.2 --iter 2  --reload_path %reload_path%_time%%i\\best_mapping.pth  --k_closest 100  --exp_id %exp_id_prefix% --src_lang %src_language% --tgt_lang %tgt_language%   --src_emb %emb_path_prefix%\\%src_language%.emb.txt --tgt_emb %emb_path_prefix%\\%tgt_language%.emb.txt



