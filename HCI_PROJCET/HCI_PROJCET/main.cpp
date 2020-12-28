#include <opencv2/opencv.hpp>
#include "opencv2/objdetect.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <windows.h>

using namespace cv;
using namespace std;

int main()
{
    //���� �޴� â ���� ���� ����
    int menu_w = 1300;
    int menu_h = 700;

    //���� �޴� â
    Mat menu(menu_h, menu_w, CV_8UC3, Scalar(0, 0, 0));

    // �÷� �̹����� ������ Mat ��ü�� �����մϴ�.
    Scalar red(0, 0, 255);
    Scalar green(0, 255, 0);
    Scalar blue(255, 0, 0);
    Scalar white(255, 255, 255);
    Scalar yellow(0, 255, 255);
    Scalar cyan(255, 255, 0);
    Scalar magenta(255, 0, 255);

    //â�� ��� ����
    int center_x = int(menu_w / 2.0);
    int center_y = int(menu_h / 2.0);

    //�ؽ�Ʈ �߰�
    int thickness = 2;
    Point location(center_x - 350, center_y - 100); //�۾� ��ġ
    int font = FONT_HERSHEY_SCRIPT_SIMPLEX;// ��Ʈ
    double fontScale = 3; //�۾� ũ��
    putText(menu, "Rock Scissor Paper", location, font, fontScale, yellow, 3); //�ؽ�Ʈ ����

    location = Point(center_x - 200, center_y + 20);
    font = FONT_ITALIC;
    fontScale = 1;
    putText(menu, "Press SPACE to start game", location, font, fontScale, red, thickness);

    location = Point(center_x - 500, center_y + 100);
    font = FONT_HERSHEY_SIMPLEX;
    fontScale = 1;
    putText(menu, "when the game start, place your palm in blue circle for 5 seconds", location, font, fontScale, blue, thickness);

    location = Point(center_x - 520, center_y + 150);
    font = FONT_HERSHEY_SIMPLEX;
    fontScale = 1;
    putText(menu, "There should be no skin color inside the red circle except your hand", location, font, fontScale, blue, thickness);

    //���α׷� ����
    while (1) {
        //�޴�â ����
        while (1) {
            imshow("Main Menu", menu);
            char ch = (char)waitKey(10);
            //space �Է½� ���� ����
            if (ch == 32) {
                destroyAllWindows(); //���� �޴� â ����
                break;
            }
            //esc �Է½� ���α׷� ����
            else if (ch == 27) { exit(0); };
        }

        //���� ���� �� �Ǵ� ����
        int rock = 0;
        int scissor = 0;
        int paper = 0;

        //���� �̹����� �ҷ��´�
        VideoCapture cam(0);

        //���� ���� �̹��� ����
        cam.set(CV_CAP_PROP_FRAME_WIDTH, 600);
        cam.set(CV_CAP_PROP_FRAME_HEIGHT, 400);
        Mat Origin_frame;

        //������ ����ũ �̹����� �ҷ���
        Mat HandMask = imread("./image/mask.png", 1);
        //ī�޶� ������ �ִ� ���(���� �޼����� ��� �� -1�� ��ȯ)
        if (!cam.isOpened()) { printf("[Error opening video cam\n"); return -1; }

        //���� �ð�
        time_t t1;
        t1 = time(NULL);

        //while������ ī�޶� �̹����� �д´�
        while (cam.read(Origin_frame)) {
            //Origin_farame �̹����� ����ִ� ���
            if (Origin_frame.empty()) {
                //���� �޼��� ��� �� ����
                printf("[No cam frame -- Break]");
                break;
            }

            //����� �������̽��� ���� Origin ����
            Mat Origin = Origin_frame.clone();
            //����� �������̽� �̹����� ���� �׸�
            circle(Origin, Point(300, 200), 120, Scalar(0, 0, 255), 0);
            circle(Origin, Point(300, 200), 50, Scalar(255, 0, 0), 0);
            //����� �������̽� ȭ�� ���
            imshow("Interface(for user)", Origin);
            //����ũ�� ����� Origin_frame�� �°� ����
            resize(HandMask, HandMask, Size(Origin_frame.cols, Origin_frame.rows), 0, 0, CV_INTER_LINEAR);

            /*����ũ ����*/
            //Origin_frame(ī�޶� �̹���)�� �ҷ��� HandMask�� ���� ����ũ�� ����
            Mat origin_mask = Origin_frame + HandMask;

            /*RGB YCrCb �𵨷� ��ȯ*/
            Mat YCrCb;//YCrCb ���� ������ ����
            //origin_mask(���� �̹����� ����ũ�� ����)�̹����� YCrCb �÷� �𵨷� ��ȯ
            cvtColor(origin_mask, YCrCb, CV_BGR2YCrCb);

            /*�Ǻ� ���� ����*/
            Mat skin_area;
            //������ �������� �ȼ� ���� ���� 0�� 255�� ��ȯ�� �Ǻ� ������ �����Ѵ�
            inRange(YCrCb, Scalar(0, 130, 90), Scalar(255, 168, 130), skin_area);

            /*�Ǻ� ���� Gray Scale �̹����� ����*/
            Mat Mask_gray; //����ũ�� ������ ī�޶� �̹����� gray scale �̹����� ������ ����
            cvtColor(origin_mask, Mask_gray, CV_BGR2GRAY); //����ũ�� ������ ī�޶� �̹����� gray scale�� ��ȯ
            //gray scale �̹����� �Ǻ� ������ ���� ������ ���� �Ǻ� ������ gray sclae�� ����(~������ �̹����� ������Ų��)
            Mat Skin_gray = Mask_gray + (~skin_area);

            //����ȭ�� ���� threshold ��
            int thresh = Skin_gray.at<uchar>(Point(300, 190)) + 50;

            /*�� ���� ����*/
            Mat thresh_skin_gray; //���� ������ ������ ����
            threshold(Skin_gray, thresh_skin_gray, thresh, 255, THRESH_BINARY); //gray scale �Ǻ� ���� ����ȭ
            Mat tild_thresh_skin_gray = ~thresh_skin_gray; //tild �������� ���� �̹��� ����
            //������ ���Ÿ� ���� erode ���� �� �ι� �����ϰ� ������ ���� ������ �ʹ� �۾����� �� �����ϱ� ���� dilate�� �ѹ� �� �־���.
            //��������� erode������ �ѹ��ϰ�, opening�� �ѹ� ������ �Ͱ� ����.
            erode(tild_thresh_skin_gray, tild_thresh_skin_gray, Mat(3, 3, CV_8U, Scalar(1)), Point(-1, -1), 2);
            erode(tild_thresh_skin_gray, tild_thresh_skin_gray, Mat(3, 3, CV_8U, Scalar(1)), Point(-1, -1), 2);
            dilate(tild_thresh_skin_gray, tild_thresh_skin_gray, Mat(3, 3, CV_8U, Scalar(1)), Point(-1, -1), 2);

            /*�Ÿ� ��ȯ �Լ� ����*/
            Mat dist;
            distanceTransform(tild_thresh_skin_gray, dist, CV_DIST_L2, 5);
            normalize(dist, dist, 255, 0, NORM_MINMAX, CV_8UC1);

            int maxIdx[2];
            int minIdx[2];
            double radius(0);

            //�չٴ� ������ �߽� ��ǥ�� ����
            minMaxIdx(dist, NULL, NULL, minIdx, maxIdx, dist);
            Point center(maxIdx[1], maxIdx[0]);
            if (center.x < 0) {
                center.x = 0;
            }
            if (center.y < 0) {
                center.y = 0;
            }
            if (radius < 0) {
                radius = 0;
            }

            radius = ((maxIdx[0] - minIdx[0]) / 2) + 10.0;//���� ������(�հ��� ������ ���� ������ ���� �������� 10.0�� ���� �־���.
            circle(Origin_frame, center, (int)(radius + 0.5), Scalar(255, 0, 0), -1); //Originf_frame�� ���� ������ �׸�
            //���α׷� Ȯ���� ���� ����� ����
            imshow("palm_area(for programmer)", Origin_frame);

            int temp = 0; //���� �ȼ� �ٷ� ���� �ȼ��� ��� ���� ����
            int count = 0; //�հ����� ���� ī��Ʈ

            /*�հ��� ���� �ľ�*/
            //�տ��� ���� �չٴ� ������ ������ ��� x ���� ���� Ž��
            for (int i = center.x - (int)radius - 20; i < center.x + (int)radius + 20; i++) {
                //i�� 0���� �۾��� ���
                if (i < 0) {
                    i = 0;
                }
                //�չٴ� ������ ã�� ��츸 ����
                else if (center.y != NULL) {
                    //���� ������ ���Ͽ� ���� ������ Ž���Ѵ�.
                    int m = center.y - (int)radius - 20;
                    //Ž���� ���� ���� �κ��� 0���� �۾����� ���
                    if (m < 0) {
                        m = 0;
                    }
                    //Ž���� �ȼ��� ��ǥ
                    Point center1(i, m);
                    //Ž���� �ȼ��� ��� ���� 255�� ���(�հ��� �κ��̴�.)
                    if (tild_thresh_skin_gray.at<uchar>(center1) == 255) {
                        //�հ��� �κ������� �ٷ� ������ �ȼ��� 255��� ����� �κ��̹Ƿ� �ϳ��� �հ������� �Ǵ�
                        //�ٷ� ������ �ȼ��� 0�̶�� ���ο� �հ����̹Ƿ� count�� 1�������� �հ����� ī��Ʈ �Ѵ�.
                        if (tild_thresh_skin_gray.at<uchar>(center1) != temp) {
                            count++;
                        }
                    }
                    //���� Ž���� �ȼ��� ���� �ٷ� ���� �ȼ� �����μ� ��� ���� ����ȴ�.
                    temp = tild_thresh_skin_gray.at<uchar>(center1);
                }
            }
            //�հ����� 0 ���� ��� rock�� ���� 1 ����
            if (count == 0) {
                rock++;
            }
            //�հ����� ������ 2���� ��� scissor�� ���� 1 ����
            else if (count == 2) {
                scissor++;
            }
            //�հ����� ������ 3�� �̻��� ��� paper�� ���� 1 ����
            else if (count >= 3) {
                paper++;
            }
            printf("%d, \n", count);

            waitKey(10);

            //���� �ð�
            time_t t2;
            t2 = time(NULL);
            //���α׷� �ð��� 5�ʰ� ���� ��
            if (t2 - t1 > 5) {
                destroyAllWindows(); //�� ī�޶� â�� �ݴ´�.
                break;
            }
        }

        //���� ���� �� ��� â
        Mat resultWindow(menu_h, menu_w, CV_8UC3, Scalar(0, 0, 0));

        //��� â �ؽ�Ʈ ����
        Point location1(center_x - 110, center_y - 200);
        font = FONT_HERSHEY_SCRIPT_SIMPLEX;// hand-writing style font
        putText(resultWindow, "Result", location1, font, 3, yellow, thickness);
        fontScale = 1;

        location1 = Point(center_x - 400, center_y - 80);
        putText(resultWindow, "PLAYER", location1, FONT_ITALIC, 2, red, thickness);

        location1 = Point(center_x + 150, center_y - 80);
        putText(resultWindow, "COMPUTER", location1, FONT_ITALIC, 2, blue, thickness);

        //5�ʵ��� �ľ��� �� rock scissor paper�� �� �� �� ���� ū ���� ����ڰ� �� �� ������� �Ѵ�.
        //(���� ������������ ����� �ֱ� ������ ������ ���� 5�ʵ��� �ݺ� Ȯ���� ��� �� ���� ���� ū ������ ����)
        int result = max(rock, scissor);
        result = max(result, paper);

        //����ڰ� �� �� ����� ������ ���
        if (result == rock) {
            printf("����");
            location1 = Point(center_x - 350, center_y + 20);
            font = FONT_ITALIC; // italic font
            fontScale = 1;
            putText(resultWindow, "Rock", location1, font, fontScale, red, thickness);
        }
        //����ڰ� �� �� ����� ������ ���
        else if (result == scissor) {
            printf("����");
            location1 = Point(center_x - 350, center_y + 20);
            font = FONT_ITALIC; // italic font
            fontScale = 1;
            putText(resultWindow, "Scissor", location1, font, fontScale, red, thickness);
        }
        //����ڰ� �� �� ����� ���� ���
        else if (result == paper) {
            printf("��");
            location1 = Point(center_x - 350, center_y + 20);
            font = FONT_ITALIC; // italic font
            fontScale = 1;
            putText(resultWindow, "Paper", location1, font, fontScale, red, thickness);
        }

        //��ǻ���� ������ ������������ ���� ���� �Լ� ���
        //0�� ���� 1�� ���� 2�� ��
        srand((unsigned int)time(NULL));
        int random = rand() % 3;
        string str = ""; //���� ���� ���� ����� ����

        //��ǻ�Ͱ� ������ �� ���
        if (random == 0) {
            location1 = Point(center_x + 300, center_y + 20);
            font = FONT_HERSHEY_SIMPLEX;  // normal size sans-serif font
            fontScale = 1;
            putText(resultWindow, "Rock", location1, font, fontScale, blue, thickness);
            if (result == rock) {
                str = "draw";
            }
            else if (result == scissor) {
                str = "lose";
            }
            else if (result == paper) {
                str = "win";
            }
        }
        //��ǻ�Ͱ� ������ �� ���
        else if (random == 1) {
            location1 = Point(center_x + 300, center_y + 20);
            font = FONT_HERSHEY_SIMPLEX;  // normal size sans-serif font
            fontScale = 1;
            putText(resultWindow, "Scissor", location1, font, fontScale, blue, thickness);
            if (result == rock) {
                str = "win";
            }
            else if (result == scissor) {
                str = "draw";
            }
            else if (result == paper) {
                str = "lose";
            }
        }
        //��ǻ�Ͱ� ���� �� ���
        else if (random == 2) {
            location1 = Point(center_x + 300, center_y + 20);
            font = FONT_HERSHEY_SIMPLEX;  // normal size sans-serif font
            fontScale == 1;
            putText(resultWindow, "Paper", location1, font, fontScale, blue, thickness);
            if (result == rock) {
                str = "lose";
            }
            else if (result == scissor) {
                str = "win";
            }
            else if (result == paper) {
                str = "draw";
            }
        }

        //���� ��� â�� ���� ��� �ؽ�Ʈ ����
        location1 = Point(center_x - 20, center_y + 200);
        font = FONT_HERSHEY_SIMPLEX;  // normal size sans-serif font
        fontScale = 1;
        putText(resultWindow, str, location1, font, fontScale, yellow, thickness);

        //���� ����, ���� �ٽ� �÷��� �ȳ��� ����
        location1 = Point(center_x - 180, center_y + 300);
        putText(resultWindow, "Press Space to Replay            Exit with [esc]", location1, FONT_ITALIC, fontScale, red, 2);
        //���� ��� â ���
        while (1) {
            imshow("Result", resultWindow);
            char ch1 = (char)waitKey(10);
            //�����̽� �Է½� �ٽ� ���� �޴��� ���ư�
            if (ch1 == 32) {
                destroyAllWindows(); //��� â ����
                break;
            }
            //ESCŰ �Է½� ���α׷� ����
            else if (ch1 == 27) { exit(0); }
        }
    }
    return 0;
}