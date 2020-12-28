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
    //시작 메뉴 창 가로 세로 길이
    int menu_w = 1300;
    int menu_h = 700;

    //시작 메뉴 창
    Mat menu(menu_h, menu_w, CV_8UC3, Scalar(0, 0, 0));

    // 컬러 이미지를 저장할 Mat 개체를 생성합니다.
    Scalar red(0, 0, 255);
    Scalar green(0, 255, 0);
    Scalar blue(255, 0, 0);
    Scalar white(255, 255, 255);
    Scalar yellow(0, 255, 255);
    Scalar cyan(255, 255, 0);
    Scalar magenta(255, 0, 255);

    //창의 가운데 저장
    int center_x = int(menu_w / 2.0);
    int center_y = int(menu_h / 2.0);

    //텍스트 추가
    int thickness = 2;
    Point location(center_x - 350, center_y - 100); //글씨 위치
    int font = FONT_HERSHEY_SCRIPT_SIMPLEX;// 폰트
    double fontScale = 3; //글씨 크기
    putText(menu, "Rock Scissor Paper", location, font, fontScale, yellow, 3); //텍스트 삽입

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

    //프로그램 시작
    while (1) {
        //메뉴창 실행
        while (1) {
            imshow("Main Menu", menu);
            char ch = (char)waitKey(10);
            //space 입력시 게임 시작
            if (ch == 32) {
                destroyAllWindows(); //메인 메뉴 창 종료
                break;
            }
            //esc 입력시 프로그램 종료
            else if (ch == 27) { exit(0); };
        }

        //가위 바위 보 판단 변수
        int rock = 0;
        int scissor = 0;
        int paper = 0;

        //비디오 이미지를 불러온다
        VideoCapture cam(0);

        //기존 원본 이미지 정의
        cam.set(CV_CAP_PROP_FRAME_WIDTH, 600);
        cam.set(CV_CAP_PROP_FRAME_HEIGHT, 400);
        Mat Origin_frame;

        //저장한 마스크 이미지를 불러옴
        Mat HandMask = imread("./image/mask.png", 1);
        //카메라에 오류가 있는 경우(에레 메세지를 출력 후 -1을 반환)
        if (!cam.isOpened()) { printf("[Error opening video cam\n"); return -1; }

        //시작 시간
        time_t t1;
        t1 = time(NULL);

        //while문으로 카메라 이미지를 읽는다
        while (cam.read(Origin_frame)) {
            //Origin_farame 이미지가 비어있는 경우
            if (Origin_frame.empty()) {
                //오류 메세지 출력 후 종료
                printf("[No cam frame -- Break]");
                break;
            }

            //사용자 인터페이스를 위한 Origin 변수
            Mat Origin = Origin_frame.clone();
            //사용자 인터페이스 이미지에 원을 그림
            circle(Origin, Point(300, 200), 120, Scalar(0, 0, 255), 0);
            circle(Origin, Point(300, 200), 50, Scalar(255, 0, 0), 0);
            //사용자 인터페이스 화면 출력
            imshow("Interface(for user)", Origin);
            //마스크의 사이즈를 Origin_frame에 맞게 조절
            resize(HandMask, HandMask, Size(Origin_frame.cols, Origin_frame.rows), 0, 0, CV_INTER_LINEAR);

            /*마스크 적용*/
            //Origin_frame(카메라 이미지)에 불러온 HandMask를 더해 마스크를 적용
            Mat origin_mask = Origin_frame + HandMask;

            /*RGB YCrCb 모델로 변환*/
            Mat YCrCb;//YCrCb 모델을 저장할 변수
            //origin_mask(원본 이미지에 마스크를 적용)이미지를 YCrCb 컬러 모델로 변환
            cvtColor(origin_mask, YCrCb, CV_BGR2YCrCb);

            /*피부 영역 추출*/
            Mat skin_area;
            //정해진 범위에서 픽셀 값을 각각 0과 255로 변환해 피부 영역을 추출한다
            inRange(YCrCb, Scalar(0, 130, 90), Scalar(255, 168, 130), skin_area);

            /*피부 영역 Gray Scale 이미지에 적용*/
            Mat Mask_gray; //마스크를 저장한 카메라 이미지의 gray scale 이미지를 저장할 변수
            cvtColor(origin_mask, Mask_gray, CV_BGR2GRAY); //마스크를 저장한 카메라 이미지를 gray scale로 변환
            //gray scale 이미지에 피부 영역의 반전 영상을 더해 피부 영역만 gray sclae로 추출(~연산은 이미지를 반전시킨다)
            Mat Skin_gray = Mask_gray + (~skin_area);

            //이진화를 위한 threshold 값
            int thresh = Skin_gray.at<uchar>(Point(300, 190)) + 50;

            /*손 영역 추출*/
            Mat thresh_skin_gray; //이진 영상을 저장할 변수
            threshold(Skin_gray, thresh_skin_gray, thresh, 255, THRESH_BINARY); //gray scale 피부 영상 이진화
            Mat tild_thresh_skin_gray = ~thresh_skin_gray; //tild 연산으로 이진 이미지 반전
            //노이즈 제거를 위해 erode 연산 을 두번 시행하고 검출한 손의 영역이 너무 작아지는 걸 방지하기 위해 dilate를 한번 해 주었다.
            //결론적으로 erode연산을 한번하고, opening을 한번 수행한 것과 같다.
            erode(tild_thresh_skin_gray, tild_thresh_skin_gray, Mat(3, 3, CV_8U, Scalar(1)), Point(-1, -1), 2);
            erode(tild_thresh_skin_gray, tild_thresh_skin_gray, Mat(3, 3, CV_8U, Scalar(1)), Point(-1, -1), 2);
            dilate(tild_thresh_skin_gray, tild_thresh_skin_gray, Mat(3, 3, CV_8U, Scalar(1)), Point(-1, -1), 2);

            /*거리 변환 함수 적용*/
            Mat dist;
            distanceTransform(tild_thresh_skin_gray, dist, CV_DIST_L2, 5);
            normalize(dist, dist, 255, 0, NORM_MINMAX, CV_8UC1);

            int maxIdx[2];
            int minIdx[2];
            double radius(0);

            //손바닥 영역의 중심 좌표를 설정
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

            radius = ((maxIdx[0] - minIdx[0]) / 2) + 10.0;//원의 반지름(손가락 검출을 위해 검출한 원의 반지름에 10.0을 더해 주었다.
            circle(Origin_frame, center, (int)(radius + 0.5), Scalar(255, 0, 0), -1); //Originf_frame에 원의 영역을 그림
            //프로그램 확인을 위해 출력한 영상
            imshow("palm_area(for programmer)", Origin_frame);

            int temp = 0; //현재 픽셀 바로 왼쪽 픽셀의 밝기 값을 저장
            int count = 0; //손가락의 개수 카운트

            /*손가락 개수 파악*/
            //앞에서 구한 손바닥 영역의 원에서 모든 x 값에 대해 탐색
            for (int i = center.x - (int)radius - 20; i < center.x + (int)radius + 20; i++) {
                //i가 0보가 작아질 경우
                if (i < 0) {
                    i = 0;
                }
                //손바닥 영역을 찾은 경우만 실행
                else if (center.y != NULL) {
                    //원의 영역에 대하여 원의 위쪽을 탐색한다.
                    int m = center.y - (int)radius - 20;
                    //탐색할 원의 위쪽 부분이 0보다 작아지는 경우
                    if (m < 0) {
                        m = 0;
                    }
                    //탐색할 픽셀의 좌표
                    Point center1(i, m);
                    //탐색할 픽셀의 밝기 값이 255인 경우(손가락 부분이다.)
                    if (tild_thresh_skin_gray.at<uchar>(center1) == 255) {
                        //손가락 부분이지만 바로 왼쪽의 픽셀도 255라면 연결된 부분이므로 하나의 손가락으로 판단
                        //바로 왼쪽의 픽셀이 0이라면 새로운 손가락이므로 count를 1증가시켜 손가락을 카운트 한다.
                        if (tild_thresh_skin_gray.at<uchar>(center1) != temp) {
                            count++;
                        }
                    }
                    //다음 탐색할 픽셀을 위해 바로 왼쪽 픽셀 값으로서 밝기 값이 저장된다.
                    temp = tild_thresh_skin_gray.at<uchar>(center1);
                }
            }
            //손가락이 0 개인 경우 rock의 개수 1 증가
            if (count == 0) {
                rock++;
            }
            //손가락의 개수가 2개인 경우 scissor의 개수 1 증가
            else if (count == 2) {
                scissor++;
            }
            //손가락의 개수가 3개 이상인 경우 paper의 개수 1 증가
            else if (count >= 3) {
                paper++;
            }
            printf("%d, \n", count);

            waitKey(10);

            //현재 시간
            time_t t2;
            t2 = time(NULL);
            //프로그램 시간이 5초가 지난 후
            if (t2 - t1 > 5) {
                destroyAllWindows(); //웹 카메라 창을 닫는다.
                break;
            }
        }

        //가위 바위 보 결과 창
        Mat resultWindow(menu_h, menu_w, CV_8UC3, Scalar(0, 0, 0));

        //결과 창 텍스트 삽입
        Point location1(center_x - 110, center_y - 200);
        font = FONT_HERSHEY_SCRIPT_SIMPLEX;// hand-writing style font
        putText(resultWindow, "Result", location1, font, 3, yellow, thickness);
        fontScale = 1;

        location1 = Point(center_x - 400, center_y - 80);
        putText(resultWindow, "PLAYER", location1, FONT_ITALIC, 2, red, thickness);

        location1 = Point(center_x + 150, center_y - 80);
        putText(resultWindow, "COMPUTER", location1, FONT_ITALIC, 2, blue, thickness);

        //5초동안 파악한 각 rock scissor paper의 개 수 중 가장 큰 값을 사용자가 낸 손 모양으로 한다.
        //(배경과 조명으로인한 노이즈가 있기 때문에 다음과 같이 5초동안 반복 확인한 결과 중 가장 값이 큰 것으로 결정)
        int result = max(rock, scissor);
        result = max(result, paper);

        //사용자가 낸 손 모양이 바위인 경우
        if (result == rock) {
            printf("바위");
            location1 = Point(center_x - 350, center_y + 20);
            font = FONT_ITALIC; // italic font
            fontScale = 1;
            putText(resultWindow, "Rock", location1, font, fontScale, red, thickness);
        }
        //사용자가 낸 손 모양이 가위인 경우
        else if (result == scissor) {
            printf("가위");
            location1 = Point(center_x - 350, center_y + 20);
            font = FONT_ITALIC; // italic font
            fontScale = 1;
            putText(resultWindow, "Scissor", location1, font, fontScale, red, thickness);
        }
        //사용자가 낸 손 모양이 보인 경우
        else if (result == paper) {
            printf("보");
            location1 = Point(center_x - 350, center_y + 20);
            font = FONT_ITALIC; // italic font
            fontScale = 1;
            putText(resultWindow, "Paper", location1, font, fontScale, red, thickness);
        }

        //컴퓨터의 랜덤한 가위바위보를 위해 랜덤 함수 사용
        //0은 바위 1은 가위 2는 보
        srand((unsigned int)time(NULL));
        int random = rand() % 3;
        string str = ""; //가위 바위 보의 결과를 저장

        //컴퓨터가 바위를 낸 경우
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
        //컴퓨터가 가위를 낸 경우
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
        //컴퓨터가 보를 낸 경우
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

        //게임 결과 창에 게임 결과 텍스트 삽입
        location1 = Point(center_x - 20, center_y + 200);
        font = FONT_HERSHEY_SIMPLEX;  // normal size sans-serif font
        fontScale = 1;
        putText(resultWindow, str, location1, font, fontScale, yellow, thickness);

        //게임 종료, 게임 다시 플레이 안내문 삽입
        location1 = Point(center_x - 180, center_y + 300);
        putText(resultWindow, "Press Space to Replay            Exit with [esc]", location1, FONT_ITALIC, fontScale, red, 2);
        //게임 결과 창 출력
        while (1) {
            imshow("Result", resultWindow);
            char ch1 = (char)waitKey(10);
            //스페이스 입력시 다시 메인 메뉴로 돌아감
            if (ch1 == 32) {
                destroyAllWindows(); //결과 창 종료
                break;
            }
            //ESC키 입력시 프로그램 종료
            else if (ch1 == 27) { exit(0); }
        }
    }
    return 0;
}