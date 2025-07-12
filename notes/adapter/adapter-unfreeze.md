        for name, param in self.named_parameters():
            # Keep adapter components trainable
            if any(component in name for component in ['spm', 'interactions', 'level_embed', 'up', 'norm1', 'norm2', 'norm3', 'norm4']):
                param.requires_grad = True
                print(f"  TRAINABLE (adapter): {name}")
            # Keep position embeddings trainable (important for adaptation)
            elif 'pos_embed' in name:
                param.requires_grad = True
                print(f"  TRAINABLE (pos_embed): {name}")
            # Keep class token trainable
            elif 'cls_token' in name:
                param.requires_grad = True
                print(f"  TRAINABLE (cls_token): {name}")
            # Keep patch embedding trainable (helps with domain adaptation)
            elif 'patch_embed' in name:
                param.requires_grad = True
                print(f"  TRAINABLE (patch_embed): {name}")
            # Keep final norm trainable
            elif name == 'norm.weight' or name == 'norm.bias':
                param.requires_grad = True
                print(f"  TRAINABLE (final_norm): {name}")
            # Freeze the transformer blocks (main DINOv2 backbone)
            elif 'blocks.' in name:
                param.requires_grad = False
                print(f"  FROZEN (backbone): {name}")
            # For anything else, default to trainable (but log it)
            else:
                param.requires_grad = True
                print(f"  TRAINABLE (other): {name}")
        # --- MODIFICATION END ---
