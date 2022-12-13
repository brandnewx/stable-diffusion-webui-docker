# syntax=docker/dockerfile:1

FROM alpine/git:2.36.2 as download
SHELL ["/bin/sh", "-ceuxo", "pipefail"]

RUN <<EOF
cat <<'EOE' > /clone.sh
mkdir -p repositories/"$1" && cd repositories/"$1" && git init && git remote add origin "$2" && git fetch origin "$3" --depth=1 && git reset --hard "$3" && rm -rf .git
EOE
EOF

RUN . /clone.sh taming-transformers https://github.com/CompVis/taming-transformers.git 24268930bf1dce879235a7fddd0b2355b84d7ea6 \
  && rm -rf data assets **/*.ipynb

RUN . /clone.sh stable-diffusion-stability-ai https://github.com/Stability-AI/stablediffusion.git 47b6b607fdd31875c9279cd2f4f16b92e4ea958e \
  && rm -rf assets data/**/*.png data/**/*.jpg data/**/*.gif

RUN . /clone.sh CodeFormer https://github.com/sczhou/CodeFormer.git c5b4593074ba6214284d6acd5f1719b6c5d739af \
  && rm -rf assets inputs

RUN . /clone.sh BLIP https://github.com/salesforce/BLIP.git 48211a1594f1321b00f14c9f7a5b4813144b2fb9
RUN . /clone.sh k-diffusion https://github.com/crowsonkb/k-diffusion.git 5b3af030dd83e0297272d861c19477735d0317ec
RUN . /clone.sh clip-interrogator https://github.com/pharmapsychotic/clip-interrogator 2486589f24165c8e3b303f84e9dbbea318df83e8


FROM brandnewx/ubuntu22:cuda-pytorch

USER root
WORKDIR /

SHELL ["/bin/bash", "-ceuxo", "pipefail"]

RUN apt-get update -qq && apt-get install -yq fonts-dejavu-core rsync git jq moreutils && apt-get clean
ENV DEBIAN_FRONTEND=noninteractive PIP_PREFER_BINARY=1 PIP_NO_CACHE_DIR=1

RUN git clone https://github.com/brandnewx/stable-diffusion-webui.git && \
  cd stable-diffusion-webui && \
  git switch merge1 && \
  git reset --hard merge1 && \
  pip install -r requirements_versions.txt

ENV ROOT=/stable-diffusion-webui

COPY --from=download /git/ ${ROOT}
RUN mkdir ${ROOT}/interrogate && cp ${ROOT}/repositories/clip-interrogator/data/* ${ROOT}/interrogate
RUN pip install -r ${ROOT}/repositories/CodeFormer/requirements.txt


ARG DEEPDANBOORU="0"
RUN [[ "${DEEPDANBOORU:-0}" == "0" ]] && : || pip install tensorflow-cpu==2.10 tensorflow-io==0.27.0 git+https://github.com/KichangKim/DeepDanbooru.git@edf73df4cdaeea2cf00e9ac08bd8a9026b7a7b26#egg=deepdanbooru

RUN pip install opencv-python-headless==4.6.0.66 \
  git+https://github.com/TencentARC/GFPGAN.git@8d2447a2d918f8eba5a4a01463fd48e45126a379 \
  git+https://github.com/openai/CLIP.git@d50d76daa670286dd6cacf3bcd80b5e4823fc8e1 \
  git+https://github.com/mlfoundations/open_clip.git@bb6e834e9c70d9c27d0dc3ecedeebeaeb1ffad6b \
  pyngrok==5.1.0
# Note: don't update the sha of previous versions because the install will take forever
# instead, update the repo state in a later step

ARG SHA=e26174a81f81ce876d037877c33795424208b091
RUN cd stable-diffusion-webui && \
  git fetch && \
  # git reset --hard ${SHA} && \
  git reset --hard && \
  pip install -r requirements_versions.txt

COPY . /docker

RUN python3 /docker/info.py ${ROOT}/modules/ui.py && \
  mv  ${ROOT}/style.css ${ROOT}/user.css && \
  sed -i 's/os.rename(tmpdir, target_dir)/shutil.move(tmpdir,target_dir)/' ${ROOT}/modules/ui_extensions.py

# Install extensions
RUN <<EOF
cat <<'EOE' > /install-extension.sh
mkdir -p ${ROOT}/extensions/"$1" && cd ${ROOT}/extensions/"$1" && git init && git remote add origin "$2" && git fetch origin "$3" --depth=1 && git reset --hard "$3" && rm -rf .git
cd ${ROOT}/extensions/"$1"
[ -f requirements.txt ] && pip install -r requirements.txt
exit 0
EOE
EOF

# RUN . /install-extension.sh sd_dreambooth_extension https://github.com/d8ahazard/sd_dreambooth_extension 725d85ca5485246a430d2f31f624646d061e8104
# RUN . /install-extension.sh stable-diffusion-webui-aesthetic-gradients https://github.com/AUTOMATIC1111/stable-diffusion-webui-aesthetic-gradients 411889ca602f20b8bb5e4d1af2b9686eab1913b1
RUN . /install-extension.sh stable-diffusion-webui-wildcards https://github.com/AUTOMATIC1111/stable-diffusion-webui-wildcards 6ed81ed1eaee9145b596ab6385c38ffd8ce7b13f
RUN . /install-extension.sh sd-dynamic-prompts https://github.com/adieyal/sd-dynamic-prompts c7434e7822bdb1d2c2f117d0edbfe28fd1f63461
RUN . /install-extension.sh stable-diffusion-webui-images-browser https://github.com/yfszzx/stable-diffusion-webui-images-browser 15528557c0434f66fd863347e2ecaff72f4f6f12
# RUN . /install-extension.sh stable-diffusion-webui-aesthetic-image-scorer https://github.com/tsngo/stable-diffusion-webui-aesthetic-image-scorer e6389da7a5827e700522ff1943ee4ae2c5010089
# RUN . /install-extension.sh auto-sd-paint-ext https://github.com/Interpause/auto-sd-paint-ext 7401bfcd05f382c56cdee147e089fc0a4ca0ea34
# RUN . /install-extension.sh a1111-sd-webui-tagcomplete https://github.com/DominikDoom/a1111-sd-webui-tagcomplete 894335f1dead68cf82ae147d3e53b37abe8fb597
# RUN . /install-extension.sh ddetailer https://github.com/dustysys/ddetailer 8e58e7fec774cfe41490a5dda6458fcbbea78493
# RUN . /install-extension.sh DreamArtist-sd-webui-extension https://github.com/7eu7d7/DreamArtist-sd-webui-extension 9e0d7da5b445d0342047c0b022baf114c8d30b33

# Copy xformers
COPY --from=brandnewx/xformers /xformers-dist /xformers-dist
RUN pip uninstall xformers --yes && pip install /xformers-dist/*.whl && rm -rf /xformers-dist

# Test initial run to install additional files.
RUN <<EOF
pip uninstall --yes \
  bitsandbytes \
  transformers \
  accelerate
pip install \
  accelerate==0.14.0 \
  bitsandbytes==0.35.0 \
  transformers==4.24.0 \
  opencv-python-headless
EOF

RUN python3 -u ${ROOT}/webui.py --test --ui-debug-mode --listen --port 7860 --ckpt-dir ${ROOT}/models/Stable-diffusion --allow-code --medvram --xformers --opt-split-attention --enable-insecure-extension-access --administrator

WORKDIR ${ROOT}
ENV CLI_ARGS=""
EXPOSE 7860
ENTRYPOINT ["/docker/entrypoint.sh"]
# run, -u to not buffer stdout / stderr
CMD python3 -u ${ROOT}/webui.py --listen --port 7860 --ckpt-dir ${ROOT}/models/Stable-diffusion ${CLI_ARGS}
