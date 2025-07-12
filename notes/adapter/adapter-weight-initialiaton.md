Looking at your dino_vit_adapter.py file, here's exactly how the adapter parameters are initialized:

Linear Layers (in _init_weights method):
```python
if isinstance(m, nn.Linear):
    trunc_normal_(m.weight, std=.02)  # Truncated normal with std=0.02
    if m.bias is not None:
        nn.init.constant_(m.bias, 0)  # Bias = 0
```

2. Conv2d/ConvTranspose2d Layers:
```python
elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
    fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    fan_out //= m.groups
    m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))  # He normal initialization
    if m.bias is not None:
        m.bias.data.zero_()  # Bias = 0
```

3. BatchNorm/LayerNorm Layers:
```python
elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
    nn.init.constant_(m.weight, 1.0)  # Weight = 1.0
    nn.init.constant_(m.bias, 0.0)    # Bias = 0.0
```

4. level_embed (in SPM):
```python
elif isinstance(m, nn.Embedding):
    m.weight.data.normal_(mean=0.0, std=1.0)  # Normal distribution for level_embed
```

So when you create a ViT Adapter model, the adapter parameters get well-designed, proper initialization (not just random), which is much better than arbitrary random initialization!

