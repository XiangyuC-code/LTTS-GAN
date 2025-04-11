# LTTS-GAN
The code of LTTS-GAN

### Abstract 
This work studies the generation of long-term time series. When used to generate long-term sequences, most existing deep learning-based generation methods suffer from
two main drawbacks, namely limited capability in exploiting long-term temporal
correlations and gradient problems during training. To tackle these obstacles, a long-term time series generative adversarial network (LTTS-GAN) is proposed in this work
by exploiting a multi-channel progressive decomposition generator. In particular, both
the generator and discriminator of the proposed LTTS-GAN are constructed upon the
transformer block. Besides, to effectively leverage the information contained within real
data, we enhance the generator's architecture by incorporating multiple channels of
trend information input, which aims to optimize the quality of the generated data.
Furthermore, the Auto-Correlation mechanism is employed to detect the period-wise
relationship while the global-local attention mechanism is proposed to balance local
and long-range information. Consequently, LTTS-GAN exhibits superior capability in
capturing long-term sequence features compared to other existing models. Finally, the
efficacy of the proposed LTTS-GAN is demonstrated through the utilization of a real-world classification application and a prediction application, which confirms the
effectiveness of the proposed model in various real-world task scenarios.

### Train
To train the GAN model:
```python
python train_GAN.py
```