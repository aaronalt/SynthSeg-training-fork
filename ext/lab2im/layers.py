def _hd95_loss(self, gt, pred, normalized_weights):
        """Differentiable HD95 surrogate evaluated ONLY for Claustrum labels (138 & 139)."""
        
        # 1. Identify Claustrum channel indices
        if hasattr(self, 'segmentation_labels') and self.segmentation_labels is not None:
            clau_mask = np.isin(self.segmentation_labels, [138, 139])
            clau_indices = np.where(clau_mask)[0]
        else:
            clau_indices = [19, 34]
        
        # 2. Slice tensors to ONLY Claustrum channels [B, X, Y, Z, 2]
        gt = tf.gather(gt, clau_indices, axis=-1)
        pred = tf.gather(pred, clau_indices, axis=-1)
        
        if normalized_weights is not None:
            normalized_weights = tf.gather(normalized_weights, clau_indices, axis=-1)
        
        # 3. FAST GT Distance Transform (Downsample 2x -> CPU EDT -> Upsample 2x)
        # 3a. Pool GT down to 96x96x96 (8x fewer voxels for SciPy to compute)
        gt_ds = tf.nn.avg_pool3d(gt, ksize=2, strides=2, padding='SAME')
        
        # 3b. Run EDT on downsampled volume on CPU
        dist_ds = tf.numpy_function(self._edt_maps, [gt_ds], tf.float32)
        dist_ds.set_shape(gt_ds.shape)
        
        # 3c. Upsample back to 192x192x192 on GPU and scale distance values by 2x factor
        dist = tf.keras.layers.UpSampling3D(size=(2, 2, 2))(dist_ds) * 2.0
        
        # 4. Soft outer edge of prediction
        n_labels = pred.get_shape().as_list()[-1]
        erosion = 1 - self.max_pooling_layer(pool_size=3, strides=1, padding='same')(1 - pred)
        soft_boundary = K.relu(pred - erosion)
        
        # 5. Flatten spatial dimensions
        s_flat = tf.reshape(soft_boundary, [tf.shape(pred)[0], -1, n_labels])
        d_flat = tf.reshape(dist, [tf.shape(pred)[0], -1, n_labels])
        
        def _hd95_batch(inputs):
            s_b, d_b = inputs
            union_score = tf.reduce_max(s_b, axis=-1)
            cand_idx_all = tf.where(union_score > 1e-3)[:, 0]
            
            cand_idx = tf.cond(
                tf.shape(cand_idx_all)[0] > 50000,
                lambda: tf.cast(tf.math.top_k(union_score, 50000).indices, 'int64'),
                lambda: cand_idx_all
            )
            
            s_sel = tf.transpose(tf.gather(s_b, cand_idx), [1, 0])
            d_sel = tf.transpose(tf.gather(d_b, cand_idx), [1, 0])
            
            num = tf.reduce_sum(s_sel * d_sel, axis=-1)
            den = tf.reduce_sum(s_sel, axis=-1) + 1e-8
            return num / den
        
        hd95 = tf.map_fn(_hd95_batch, (s_flat, d_flat), dtype='float32')
        
        # 6. Channel weighting and batch reduction
        if normalized_weights is not None:
            weights = normalized_weights / (tf.reduce_sum(normalized_weights, -1, keepdims=True) + 1e-8)
            hd95 = tf.reduce_sum(hd95 * weights, -1)
        else:
            hd95 = tf.reduce_mean(hd95, axis=-1)
        
        return tf.math.reduce_mean(hd95)