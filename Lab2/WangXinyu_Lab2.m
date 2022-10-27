%% Load target folder
cd /Users/ricardo/Desktop/CE4003/Lab2

%% Q3.1 Edge Detection
%% Part a
Img = imread('macritchie.jpg');
Img = rgb2gray(Img);
subplot(1,3,1);
imshow(Img);
title('GreyScale Image');

%% Part b
vertical_sobel = [-1 0 1;-2 0 2;-1 0 1];
horizontal_sobel = [-1 -2 -1;0 0 0;1 2 1];

Img_vertical_filtered = conv2(double(Img),double(vertical_sobel));
Img_horizontal_filtered = conv2(double(Img),double(horizontal_sobel));

subplot(1,3,1);
imshow(Img);
title('GreyScale Image');
subplot(1,3,2);
imshow(Img_vertical_filtered);
title('Vertical filtered Image');
subplot(1,3,3);
imshow(Img_horizontal_filtered);
title('Horizontal filtered Image');

%% Part c
Img_combined = sqrt(Img_horizontal_filtered.^2 + Img_vertical_filtered.^2);
imshow(Img_combined);
title('Combined Edge Image')

%% Part d
img_100 = Img_combined>100;
img_150 = Img_combined>150;
img_200 = Img_combined>200;
subplot(1,3,1);imshow(img_100);title('threshold=100');
subplot(1,3,2);imshow(img_150);title('threshold=150');
subplot(1,3,3);imshow(img_200);title('threshold=200');


%% Part e
Img_canny = edge(Img,'canny',[0.04,0.1],1.0);
subplot(1,3,1);imshow(Img_canny);title('sigma=1.0');

%Part_i
Img_canny_1 = edge(Img,'canny',[0.04,0.1],4.0);
subplot(1,3,2);imshow(Img_canny_1);title('sigma=4.0');
%The larger the sigma value is, the more noisy edgel is removed.
%But as the sigma value increase, the less its location is accurate.

%Part_ii
Img_canny_2 = edge(Img,'canny',[0.08,0.1],4.0);
subplot(1,3,3);imshow(Img_canny_2);title('tl=0.08,sigma=4.0');
%The smaller the tl value is, the more short discrete lines segments are
%allowed.

%% Q3.2 Line Finding using Hough Transform
%% Part a
Img_canny = edge(Img,'canny',[0.04,0.1],1.0);
imshow(Img_canny);title('original canny image');
%% Part b
theta = 0:180;
[H,xp] = radon(Img_canny);
imagesc(H);
xlabel('\theta (degrees)');
ylabel('\rho');
colormap(gca,hot), colorbar;

%% Part c
max_intensity = max(max(H));
[radial_distance_max,theta_max] = find(max_intensity==H);
fprintf('the coordinates of max intensity point:\n');
fprintf('radius= %d , theta= %d\n',radial_distance_max,theta_max);
%% Part d
radius = xp(radial_distance_max); %what xp and radius relation is?
[A,B] = pol2cart(theta_max*pi/180,radius);
[height,width] = size(Img);
B = -B;
C = A*(A+width/2)+B*(B+height/2);
fprintf('Part d Results:\n');
fprintf('A = %6.4f, B = %6.4f, C = %6.4f\n',A,B,C);

%% Part e
xl = 0;
yl = (C-A*xl)/B;
[height,width] = size(Img);
xr = width-1;
yr = (C-A*xr)/B;
fprintf('Part e Results:\n');
fprintf('yl = %6.4f, yr = %6.4f\n',yl,yr);

%% Part f
imshow(Img);
line([xl xr],[yl yr],'Color','Red');

%% Q3.3 3D Stereo
%% Part a
