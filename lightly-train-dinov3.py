import lightly_train

if __name__ == "__main__":
    lightly_train.train(
        out="out/my_experiment",
        overwrite=True,
        data="my_data_dir",
        model="torchvision/resnet18",
        method="distillationv1",
        method_args={
            "teacher": "dinov3/vits16",
            "teacher_url":"https://gitclone.com/download1/aliendao/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
        }
    )
