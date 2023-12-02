% Myrsini Gkolemi

% 1.
% Padding options:
% circular: pads with circular repetition of elements
% replicate: repeats border elements of A
% symmetric: pads array with mirror reflections of itself
% numeric constant: e.g. 1
% The default filter size is 2*ceil(2*sigma) + 1.

bsds = imread('img\task_03\bsds86016.png');
bsds_binarize = imbinarize(bsds);
subplot(1, 2, 1), imshow(bsds);
title('Original Image');
subplot(1, 2, 2), imshow(bsds_binarize);
title('Binarized Image');
suptitle('1. Binarization without filter');
% saveas(gcf, "bin_img.jpg");
sigma = [3, 5, 7];
% default padding = replicate
padding_options = cell(1, 4);
padding_options{1} = 0;
padding_options{2} = 'circular';
padding_options{3} = 'replicate';
padding_options{4} = 'symmetric';
gaussian_blur_imgs = cell(1, 3);
binary_images = cell(1, 3);

for i = 1:4 % padding = circular, replicate, symmetric
    if size(padding_options{1}) == 1
        method_title = num2str(padding_options{i});
    else
        method_title = padding_options{i};
    end

    figure('Name', ['Gaussian Blur with ', method_title, ' Padding'])

    for j = 1:3 % sigma = 3,5,7
        gaussian_blur_imgs{j} = imgaussfilt(bsds, sigma(j), 'Padding', ...
            padding_options{i});
        subplot(2, 3, j), imshow(gaussian_blur_imgs{j});
        title(['Gaussian blurred image sigma = ', num2str(sigma(j))]);
        binary_images{j} = imbinarize(gaussian_blur_imgs{j});
        subplot(2, 3, j + 3), imshow(binary_images{j});
        title(['Binary image sigma = ', num2str(sigma(j))]);
    end

    suptitle(['1. Gaussian Blur with ', method_title, ' Padding']);
    %     filename = [num2str(i),'_gauss_blur_pad.jpg'];
    %     saveas(gcf, filename);
end

% 2.
% Average filter with size 5 x 5
% 5 x 5

facade = imread('img\task_03\facade.png');
% imfilter by default uses corellation, fspecial uses correlation kernels
filtered_img_2 = imfilter(facade, fspecial('average', [5 5]));
bin_img_2 = imbinarize(filtered_img_2);
figure('Name', '2. Binarization after Average 5x5 filter');
subplot(1, 2, 1), imshow(facade);
title('Original Image');
subplot(1, 2, 2), imshow(bin_img_2);
title('Binarized Image');
suptitle('2. Binarization after Average 5x5 filter');
% saveas(gcf, 'average_binarize.jpg');

% 3.
% G1 - G2
% G1: stdev = 1.28*S when G2:S
% 1 sqrt(2) 2 2*sqrt(2)
% binarize

plate_usa = imread('img'\task_03\plate_usa.png');
bin_img_3 = imbinarize(plate_usa);
sigma = [1, sqrt(2), 2 2 * sqrt(2)];
laplace_filters = cell(4, 1);
bin_laplace_imgs = cell(4, 1);

figure('Name', '3. Original Image & Binarized Image');
subplot(1, 2, 1), imshow(plate_usa);
title("Original Image");
subplot(1, 2, 2), imshow(bin_img_3);
title("Binarized Image");
suptitle('2. Binarization without filter');
figure('Name', '3. Laplace Filtered & Binarized Image');

for i = 1:4
    laplace_filters{i} = imgaussfilt(plate_usa, 1.28 * sigma(i), ...
        'Padding', 'circular') - imgaussfilt(plate_usa, sigma(i), ...
        'Padding', 'circular');
    subplot(2, 4, i), imshow(laplace_filters{i});
    title(['Laplace filtered image sigma = ', num2str(sigma(i))]);
    bin_laplace_imgs{i} = imbinarize(laplace_filters{i});
    subplot(2, 4, i + 4), imshow(bin_laplace_imgs{i});
    title(['Binarized filtered image sigma = ', num2str(sigma(i))]);
end

suptitle('3. Laplace Filtered & Binarized Image');
% saveas(gcf, 'laplace.jpg');
