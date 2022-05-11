# python main_test_clustertransformer.py --benchmark div2k --task swin_sr --model_path nsmltrained_models/KR80934_CVLAB_SR_54/250000/model/G.pth --scale 2
# python main_test_clustertransformer.py --benchmark Set5 --task gumbel_simple_small_yes --model_path nsmltrained_models/KR80934_CVLAB_SR2_23/250000/model/G.pth
# python main_test_clustertransformer.py --benchmark div2k --task gumbel_toosimple_small_yes --model_path nsmltrained_models/KR80934_CVLAB_SR2_26/250000/model/G.pth

# python main_test_clustertransformer.py --benchmark Set5 --task gumbel_simple_small_yes --model_path nsmltrained_models/KR80934_CVLAB_SR2_23/250000/model/G.pth
# python main_test_clustertransformer.py --benchmark div2k --task compensate1 --model_path nsmltrained_models/KR80934_CVLAB_SR2_65/250000/model/G.pth
python main_test_clustertransformer.py --benchmark div2k --task compensate2 --model_path nsmltrained_models/KR80934_CVLAB_SR2_66/250000/model/G.pth
# python main_test_clustertransformer.py --benchmark div2k --task compensate3 --model_path nsmltrained_models/KR80934_CVLAB_SR2_68/210000/model/G.pth


### Old One
# python main_test_clustertransformer.py --benchmark div2k --task swin_sr --model_path nsmltrained_models/KR80934_CVLAB_SR_54/250000/model/G.pth
# python main_test_clustertransformer.py --benchmark div2k --task noswin_sr --model_path nsmltrained_models/KR80934_CVLAB_SR_56/250000/model/G.pth
# python main_test_clustertransformer.py --benchmark div2k --task onlySA_sr --model_path nsmltrained_models/KR80934_CVLAB_SR_200/250000/model/G.pth
# python main_test_clustertransformer.py --benchmark div2k --task kmeans_sr --model_path nsmltrained_models/KR80934_CVLAB_SR_264/95000/model/G.pth
# python main_test_clustertransformer.py --benchmark div2k --task kmeans_post_sr --model_path nsmltrained_models/KR80934_CVLAB_SR_297/250000/model/G.pth
# python main_test_clustertransformer.py --benchmark div2k --task kmeans_last_sr --model_path nsmltrained_models/KR80934_CVLAB_SR_298/250000/model/G.pth
# python main_test_clustertransformer.py --benchmark div2k --task kmeans_postkeepv_sr --model_path nsmltrained_models/KR80934_CVLAB_SR_288/465000/model/G.pth
# python main_test_clustertransformer.py --benchmark div2k --task kmeans_postkeepvnorecycle_sr --model_path nsmltrained_models/KR80934_CVLAB_SR_290/500000/model/G.pth
# python main_test_clustertransformer.py --benchmark div2k --task blockcluster_noswin_sr --model_path nsmltrained_models/KR80934_CVLAB_SR_330/250000/model/G.pth
# python main_test_clustertransformer.py --benchmark div2k --task random_noswin_sr --model_path nsmltrained_models/KR80934_CVLAB_SR_334/250000/model/G.pth
# python main_test_clustertransformer.py --benchmark Set5 --task randomfix_noswin_sr --model_path nsmltrained_models/KR80934_CVLAB_SR_339/250000/model/G.pth
# python main_test_clustertransformer.py --benchmark Set14 --task randomfix_noswin_sr --model_path nsmltrained_models/KR80934_CVLAB_SR_339/250000/model/G.pth
# python main_test_clustertransformer.py --benchmark B100 --task randomfix_noswin_sr --model_path nsmltrained_models/KR80934_CVLAB_SR_339/250000/model/G.pth
# python main_test_clustertransformer.py --benchmark Urban100 --task randomfix_noswin_sr --model_path nsmltrained_models/KR80934_CVLAB_SR_339/250000/model/G.pth
# python main_test_clustertransformer.py --benchmark div2k --task intrakmeans_post_sr --model_path nsmltrained_models/KR80934_CVLAB_SR_346/195000/model/G.pth
# python main_test_clustertransformer.py --benchmark div2k --task kmeans_normpost_sr --model_path nsmltrained_models/KR80934_CVLAB_SR_385/250000/model/G.pth
# python main_test_clustertransformer.py --benchmark div2k --task halfkmeans_post_sr --model_path nsmltrained_models/KR80934_CVLAB_SR_390/250000/model/G.pth
# python main_test_clustertransformer.py --benchmark div2k --task kmeans_postkeepv_bias_sr --model_path nsmltrained_models/KR80934_CVLAB_SR_397/230000/model/G.pth
# python main_test_clustertransformer.py --benchmark div2k --task kmeans_tooshort_sr --model_path nsmltrained_models/KR80934_CVLAB_SR_404/245000/model/G.pth
# python main_test_clustertransformer.py --benchmark div2k --task kmeans_tooshortkeepv_sr --model_path nsmltrained_models/KR80934_CVLAB_SR_406/190000/model/G.pth