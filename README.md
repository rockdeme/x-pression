# X-Pression: deep learning-based framework for 3D gene expression programs prediction

**X-Pression** is a deep learning-based framework that enables scalable and cost-effective inference of 3D gene expression programs from micro-CT data.

## Key Points:
- **3D Gene Expression Prediction**: Leverages micro-CT imaging and spatial transcriptomics to predict gene expression at a tissue and organ scale.
- **Scalable & Cost-Effective**: Utilizes low-cost, non-destructive micro-CT imaging for large-scale studies.
- **Integration with ST**: Combines micro-CT scans with spatial transcriptomics for enhanced tissue analysis.
- **Versatility**: Applicable to a range of diseases and imaging modalities, including cancer, infections, and neurodegenerative disorders.

## Reference:
[Publication: Deep learning-based 3D spatial transcriptomics with X-Pression](#)

### Deep learning-based 3D spatial transcriptomics with X-Pression

Demeter Túrós, Lollija Gladiseva, Marius Botos, Chang He, G. Tuba Barut, Inês Berenguer Veiga, Nadine Ebert, Anne Bonnin, Astrid Chanfon, Llorenç Grau-Roma, Alberto Valdeolivas, Sven Rottenberg, Volker Thiel, and Etori Aguiar Moreira.

## Abstract
Spatial transcriptomics technologies currently lack scalable and cost-effective options to profile tissues in three dimensions. Technological advances in microcomputed tomography enable non-destructive volumetric imaging of tissue blocks with sub-micron resolution at a centimetre scale. Here, we present **X-Pression**, a deep convolutional neural network-based framework designed to impute 3D expression signatures of cellular niches to microcomputed tomography volumetric data. By training on a single 2D section of paired spatial transcriptomics data, **X-Pression** achieves high accuracy and is able to generalise on out-of-sample examples.  

We utilised **X-Pression** to demonstrate the benefit of 3D examination of tissues on using a SARS-CoV-2 vaccine efficacy spatial transcriptomics and microcomputed tomography cohort of a recently developed live attenuated SARS-CoV-2 vaccine. Through the application of **X-Pression** to the entire mouse lung, we visualised the sites of viral replication at the organ level and the simultaneous collapse of small alveoli in their vicinity. We also assessed the immunological response following vaccination and virus challenge infection.  

**X-Pression** offers a valuable and cost-effective addition to infer expression signatures without the need for consecutive 2D sectioning and reconstruction, providing new insights into transcriptomic signatures in three dimensions.
