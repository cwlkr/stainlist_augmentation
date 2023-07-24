# Stain List Augmentation

Stain augmentation pipeline employing randomized stain mixup of batched data with a list of TCGA WSI slides with different H&E staining. 

Idea of this augmention is to improve model generalizability by opening up the space of learned color reprentaions.

See also [stain_mixup](https://github.com/aetherAI/stain-mixup/tree/main/stain_mixup).

## Usage

```
import stainmixup_albmt as salbmt

img_transform = albmt.Compose([
	salbmt.StainMix(salbmt.load_stain_mix_matrix(), salbmt.load_norm_source(), p = 0.8),
    ToTensor()
    ])
```

At the moment load_norm_source is loading an image named 'reference.png' from the stain_mixup folder.

## List of TCGA images used

TCGA-22-5471-01Z-00-DX1.AACEB098-E9B8-4A2B-905E-7D66BE962922.svs    
TCGA-33-4547-01Z-00-DX7.91be6f90-d9ab-4345-a3bd-91805d9761b9.svs   
TCGA-50-6594-01Z-00-DX1.43b2005a-4245-4025-ad85-4a957f308a5c.svs     
TCGA-95-8494-01Z-00-DX1.716299EF-71BB-4095-8F4D-F0C2252CE594.svs   
TCGA-95-A4VK-01Z-00-DX1.D09778E0-285E-4593-84C8-B6009DDF4E41.svs   
TCGA-AA-3675-01A-01-BS1.9a262926-cfce-4642-864a-d670acd672d6.svs   
TCGA-AA-3866-01Z-00-DX1.f93457c3-abaa-4268-84e2-394d7c1aa523.svs   
TCGA-AA-3973-01Z-00-DX1.05cee752-3f4e-442d-a093-dcfb2b6130f0.svs   
