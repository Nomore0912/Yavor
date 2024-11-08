#!/usr/bin/env python
# -*- coding:utf-8 -*-
from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    T5EncoderModel,
    T5Tokenizer,
)
import torch
from models.SD3 import SD3Transformer
from diffusers.image_processor import VaeImageProcessor
from diffusers import AutoencoderKL


class StableDiffusion3(object):

    def __init__(self):
        self.vae = AutoencoderKL()
        self.image_processor = VaeImageProcessor(vae_scale_factor=8)
        pass

    @staticmethod
    def _get_clip_prompt_embeds(
            prompt,
            device,
            clip_model_index,
            num_images_per_prompt: int = 1,
    ):
        text_encoder_1 = CLIPTextModelWithProjection.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")
        tokenizer_1 = CLIPTokenizer.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")
        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-large-patch14")
        tokenizer_2 = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

        clip_tokenizers = [tokenizer_1, tokenizer_2]
        clip_text_encoders = [text_encoder_1, text_encoder_2]
        tokenizer = clip_tokenizers[clip_model_index]
        text_encoder = clip_text_encoders[clip_model_index]

        batch_size = len(prompt)
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)
        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds.hidden_states[-2]
        prompt_embeds = prompt_embeds.to(dtype=text_encoder_1.dtype, device=device)
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt, 1)
        pooled_prompt_embeds = pooled_prompt_embeds.view(batch_size * num_images_per_prompt, -1)
        return prompt_embeds, pooled_prompt_embeds

    @staticmethod
    def _get_t5_prompt_embeds(
            prompt,
            num_images_per_prompt: int = 1,
            max_sequence_length: int = 256,
            device=None,
            dtype=None,
    ):
        batch_size = len(prompt)
        tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-xxl")
        text_encoder_3 = T5EncoderModel.from_pretrained("google/t5-v1_1-xxl")
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt")
        text_input_ids = text_inputs.input_ids
        prompt_embeds = text_encoder_3(text_input_ids.to(device))[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
        return prompt_embeds

    def encoder(
            self,
            prompt,
            prompt_2,
            prompt_3,
            device,
            prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
            do_classifier_free_guidance=True,
            negative_prompt=None,
            negative_prompt_2=None,
            negative_prompt_3=None,
            negative_prompt_embeds=None,
    ):
        if prompt_embeds is None:
            prompt_2 = prompt_2 or prompt
            prompt_3 = prompt_3 or prompt

            # OpenCLIP-bigG-14
            prompt_embed, pooled_prompt_embed = self._get_clip_prompt_embeds(
                prompt=prompt,
                device=device,
                num_images_per_prompt=1,
                clip_model_index=0,
            )
            # OpenAIClip-l-14
            prompt_2_embed, pooled_prompt_2_embed = self._get_clip_prompt_embeds(
                prompt=prompt_2,
                device=device,
                num_images_per_prompt=1,
                clip_model_index=1,
            )
            clip_prompt_embeds = torch.cat([prompt_embed, prompt_2_embed], dim=-1)

            # T5-XXL
            t5_prompt_embed = self._get_t5_prompt_embeds(
                prompt=prompt_3,
                num_images_per_prompt=1,
                max_sequence_length=256,
                device=device,
            )

            clip_prompt_embeds = torch.nn.functional.pad(
                clip_prompt_embeds, (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1])
            )
            prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)
            pooled_prompt_embeds = torch.cat([pooled_prompt_embed, pooled_prompt_2_embed], dim=-1)
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt_2 = negative_prompt_2 or negative_prompt
            negative_prompt_3 = negative_prompt_3 or negative_prompt

            negative_prompt_embed, negative_pooled_prompt_embed = self._get_clip_prompt_embeds(
                negative_prompt,
                device=device,
                num_images_per_prompt=1,
                clip_model_index=0,
            )
            negative_prompt_2_embed, negative_pooled_prompt_2_embed = self._get_clip_prompt_embeds(
                negative_prompt_2,
                device=device,
                num_images_per_prompt=1,
                clip_model_index=1,
            )
            negative_clip_prompt_embeds = torch.cat([negative_prompt_embed, negative_prompt_2_embed], dim=-1)

            t5_negative_prompt_embed = self._get_t5_prompt_embeds(
                prompt=negative_prompt_3,
                num_images_per_prompt=1,
                max_sequence_length=256,
                device=device,
            )

            negative_clip_prompt_embeds = torch.nn.functional.pad(
                negative_clip_prompt_embeds,
                (0, t5_negative_prompt_embed.shape[-1] - negative_clip_prompt_embeds.shape[-1]),
            )

            negative_prompt_embeds = torch.cat([negative_clip_prompt_embeds, t5_negative_prompt_embed], dim=-2)
            negative_pooled_prompt_embeds = torch.cat(
                [negative_pooled_prompt_embed, negative_pooled_prompt_2_embed], dim=-1
            )
        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

    @staticmethod
    def prepare_latents(
            batch_size,
            num_channels_latents,
            height,
            width,
            dtype,
            device,
            generator,
            latents=None,
    ):
        if latents is not None:
            return latents.to(dtype=dtype, device=device)

        shape = (
            batch_size,
            num_channels_latents,
            int(height) // 8,
            int(width) // 8,
        )

        # rand_n tensor
        batch_size_rand = shape[0]
        layout = torch.strided

        if isinstance(generator, list) and len(generator) == 1:
            generator = generator[0]

        if isinstance(generator, list):
            shape = (1,) + shape[1:]
            latents = [
                torch.randn(shape, generator=generator[i], device=device, dtype=dtype, layout=layout)
                for i in range(batch_size_rand)
            ]
            latents = torch.cat(latents, dim=0).to(device)
        else:
            latents = torch.randn(shape, generator=generator, device=device, dtype=dtype, layout=layout).to(device)

        return latents

    def main(
            self,
            prompt,
            prompt_2,
            prompt_3,
            height,
            width,
            num_inference_steps=28,
            guidance_scale=7.0,
            negative_prompt=None,
            negative_prompt_2=None,
            negative_prompt_3=None,
            num_images_per_prompt=1,
            generator=None,
            latents=None,
            scheduler=None,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            pooled_prompt_embeds=None,
            negative_pooled_prompt_embeds=None,
    ):
        batch_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encoder(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_3=prompt_3,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            do_classifier_free_guidance=True,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            device=device,
        )

        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

        time_steps, num_inference_steps = retrieve_time_steps(scheduler, num_inference_steps, device)
        _num_time_steps = len(time_steps)

        num_channels_latents = 16
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        transformer = SD3Transformer()

        for i, t in enumerate(time_steps):
            latent_model_input = torch.cat([latents] * 2)
            timestep = t.expand(latent_model_input.shape[0])
            noise_pred = transformer(
                hidden_states=latent_model_input,
                time_step=timestep,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
            )[0]
            noise_pred_un_cond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_un_cond + guidance_scale * (noise_pred_text - noise_pred_un_cond)
            latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        latents = (latents / 0.18215)
        image = self.vae.decode(latents, return_dict=False)[0]
        image = self.image_processor.postprocess(image, output_type="pil")
        return image


def retrieve_time_steps(
    scheduler,
    num_inference_steps=None,
    device=None,
    **kwargs,
):
    scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
    time_steps = scheduler.timesteps
    return time_steps, num_inference_steps
