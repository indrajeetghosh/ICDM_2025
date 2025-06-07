# ICDM_2025

Official Implementation of MemGaze Framework


**Abstract:-** Image memorability (IM) estimation typically relies on learning generic semantic features from large-scale datasets;
however, memorability is intrinsically individual-dependent and personalized visual viewing behavior, but overlooking this variability can undermine model performance in downstream cognitive and human–machine interaction tasks, leading towards suboptimal performance. To address this, we propose MemGaze, a unified framework that integrates knowledge distillation (KD) and imitation learning (IL) to jointly learn generic and personalized salient representations for memorability estimation by leveraging both image content and gaze-derived heatmaps. MemGaze employs a teacher network built upon a pretrained ResNet-50 backbone, followed by an encoder–decoder archi-
tecture coupled with spatial and channel attention mechanisms to generate generic saliency-aware memorability maps. A lightweight, attention-guided student encoder–decoder is then optimized through a composite imitation-guided distillation process, where knowledge is distilled from the teacher while simultaneously imitating user-specific gaze fixation heatmaps. Through this joint training process, the student network learns to produce personalized and accurate memorability estimates while achieving substantial reductions in computational complexity. We validate MemGaze on two public image memorability estimation datasets (LaMem and SUN) and a novel in-house dataset comprising 45 participants engaged in visual search and navigation tasks that reflect individualized visual attention patterns. MemGaze outperforms nine state-of-the-art IM models by capturing coarse-to-fine saliency and adapting to individual attention, achieving a 6% relative improvement in memorability prediction


**Overall Pipeline:-**

![MemGaze Framework comprises three modules: (i) a \emph{semantic feature extraction module} leveraging pretrained visual representations, (ii) an \emph{attention-based teacher--student architecture} for learning generic semantic memorability patterns, and (iii) a \emph{user-conditioned personalization module} optimized via a joint knowledge distillation and imitation learning~(behavioral cloning) objective.](MemGaze.png)
