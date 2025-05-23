import os
import cv2
import argparse
import numpy as np
from hivision.error import FaceError
from hivision.utils import (
    hex_to_rgb,
    resize_image_to_kb,
    add_background,
    save_image_dpi_to_bytes,
)
from hivision import IDCreator
from hivision.creator.layout_calculator import (
    generate_layout_array,
    generate_layout_image,
)
from hivision.creator.choose_handler import choose_handler

INFERENCE_TYPE = [
    "idphoto",
    "human_matting",
    "add_background",
    "generate_layout_photos",
    "idphoto_crop",
]
MATTING_MODEL = [
    "hivision_modnet",
    "modnet_photographic_portrait_matting",
    "mnn_hivision_modnet",
    "rmbg-1.4",
    "birefnet-v1-lite",
]
FACE_DETECT_MODEL = [
    "mtcnn",
    "face_plusplus",
    "retinaface-resnet50",
]
RENDER = [0, 1, 2]


def run(
    input_image_path,
    output_image_path,
    type="idphoto",
    height=413,
    width=295,
    color="638cce",
    hd=True,
    kb=None,
    render=0,
    dpi=300,
    face_align=False,
    matting_model="modnet_photographic_portrait_matting",
    face_detect_model="mtcnn",
):
    creator = IDCreator()
    choose_handler(creator, matting_model, face_detect_model)

    input_image = cv2.imread(input_image_path, cv2.IMREAD_UNCHANGED)

    size = (int(height), int(width))

    if type == "idphoto":
        try:
            result = creator(input_image, size=size, face_alignment=face_align)
        except FaceError:
            print("人脸数量不等于 1，请上传单张人脸的图像。")
            return
        # save_image_dpi_to_bytes(
        #     cv2.cvtColor(result.standard, cv2.COLOR_RGBA2BGRA),
        #     output_image_path,
        #     dpi=dpi,
        # )
        file_name, file_extension = os.path.splitext(output_image_path)
        new_file_name = file_name + "_hd" + file_extension
        save_image_dpi_to_bytes(
            cv2.cvtColor(result.hd, cv2.COLOR_RGBA2BGRA), new_file_name, dpi=dpi
        )

    elif type == "human_matting":
        result = creator(input_image, change_bg_only=True)
        cv2.imwrite(output_image_path, result.hd)

    elif type == "add_background":
        render_choice = ["pure_color", "updown_gradient", "center_gradient"]
        rgb_color = hex_to_rgb(color)
        bgr_color = (rgb_color[2], rgb_color[1], rgb_color[0])
        print(f"Applying background color: BGR{bgr_color}")

        # Initialize IDCreator for matting
        creator = IDCreator()
        choose_handler(creator, matting_model, face_detect_model)

        # Perform human matting
        input_image = cv2.imread(input_image_path, cv2.IMREAD_UNCHANGED)
        result = creator(input_image, change_bg_only=True)
        matted_image = result.hd  # Use HD matted output (RGBA)

        # Apply background color
        result_image = add_background(
            matted_image, bgr=bgr_color, mode=render_choice[render]
        )
        result_image = result_image.astype(np.uint8)

        # Ensure no alpha channel (convert to BGR for solid background)
        if result_image.shape[2] == 4:
            result_image = cv2.cvtColor(result_image, cv2.COLOR_RGBA2BGR)
        else:
            result_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)

        # Force JPEG output
        output_image_path = os.path.splitext(output_image_path)[0] + ".jpg"

        if kb:
            resize_image_to_kb(result_image, output_image_path, int(kb), dpi=dpi)
        else:
            save_image_dpi_to_bytes(result_image, output_image_path, dpi=dpi)

    elif type == "generate_layout_photos":
        typography_arr, typography_rotate = generate_layout_array(
            input_height=size[0], input_width=size[1]
        )
        result_layout_image = generate_layout_image(
            input_image,
            typography_arr,
            typography_rotate,
            height=size[0],
            width=size[1],
        )

        if kb:
            result_layout_image = cv2.cvtColor(result_layout_image, cv2.COLOR_RGB2BGR)
            resize_image_to_kb(result_layout_image, output_image_path, int(kb), dpi=dpi)
        else:
            save_image_dpi_to_bytes(
                cv2.cvtColor(result_layout_image, cv2.COLOR_RGBA2BGRA),
                output_image_path,
                dpi=dpi,
            )

    elif type == "idphoto_crop":
        try:
            result = creator(input_image, size=size, crop_only=True)
        except FaceError:
            print("人脸数量不等于 1，请上传单张人脸的图像。")
            return
        save_image_dpi_to_bytes(
            cv2.cvtColor(result.standard, cv2.COLOR_RGBA2BGRA),
            output_image_path,
            dpi=dpi,
        )
        file_name, file_extension = os.path.splitext(output_image_path)
        new_file_name = file_name + "_hd" + file_extension
        save_image_dpi_to_bytes(
            cv2.cvtColor(result.hd, cv2.COLOR_RGBA2BGRA), new_file_name, dpi=dpi
        )


def main():
    parser = argparse.ArgumentParser(
        description="HivisionIDPhotos 证件照制作推理程序。"
    )
    parser.add_argument(
        "-t",
        "--type",
        help="请求 API 的种类",
        choices=INFERENCE_TYPE,
        default="idphoto",
    )
    parser.add_argument("-i", "--input_image_dir", help="输入图像路径", required=True)
    parser.add_argument("-o", "--output_image_dir", help="保存图像路径", required=True)
    parser.add_argument("--height", help="证件照尺寸-高", default=413)
    parser.add_argument("--width", help="证件照尺寸-宽", default=295)
    parser.add_argument("-c", "--color", help="证件照背景色", default="638cce")
    parser.add_argument("--hd", type=bool, help="是否输出高清照", default=True)
    parser.add_argument(
        "-k", "--kb", help="输出照片的 KB 值，仅对换底和制作排版照生效", default=None
    )
    parser.add_argument(
        "-r",
        "--render",
        type=int,
        help="底色合成的模式，有 0:纯色、1:上下渐变、2:中心渐变 可选",
        choices=RENDER,
        default=0,
    )
    parser.add_argument("--dpi", type=int, help="输出照片的 DPI 值", default=300)
    parser.add_argument(
        "--face_align", type=bool, help="是否进行人脸旋转矫正", default=False
    )
    parser.add_argument(
        "--matting_model",
        help="抠图模型权重",
        default="modnet_photographic_portrait_matting",
        choices=MATTING_MODEL,
    )
    parser.add_argument(
        "--face_detect_model",
        help="人脸检测模型",
        default="mtcnn",
        choices=FACE_DETECT_MODEL,
    )

    args = parser.parse_args()

    run(
        input_image_path=args.input_image_dir,
        output_image_path=args.output_image_dir,
        type=args.type,
        height=args.height,
        width=args.width,
        color=args.color,
        hd=args.hd,
        kb=args.kb,
        render=args.render,
        dpi=args.dpi,
        face_align=args.face_align,
        matting_model=args.matting_model,
        face_detect_model=args.face_detect_model,
    )


if __name__ == "__main__":
    main()


#working!!