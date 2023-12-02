% Myrsini Gkolemi

range = -5:1:5;
h = [0.009 0.027 0.065 0.122 0.177 0.2 0.177 0.122 0.065 0.027 0.009];
g = [0.013 0.028 0.048 0.056 0.039 0 -0.039 -0.056 -0.048 -0.028 -0.013];
filter_size = length(range);

figure('Name', '1. h,g filters in time (s)');
plot(range, h);
hold on;
plot(range, g);
hold off;
legend('h', 'g');
title('1. h,g filters in time (s)');

% Save figure
filename = 'filters_in_time.jpg';
saveas(gcf, filename);

syms u; syms v;

Hu = 0;
Gu = 0;

for n = 1:filter_size
    Hu = Hu + h(n) * exp(-j * 2 * pi * u * n);
    Gu = Gu + g(n) * exp(-j * 2 * pi * v * n);
end

figure('Name', '2. Magnitude of Fourier Transform for filters h, g');
fplot(abs(Hu), [-1/2, 1/2]);
hold on;
fplot(abs(Gu), [-1/2, 1/2]);
hold off;
title('2. Magnitude of Fourier Transform for filters h, g');
legend('|Hu|', '|Gu|');

% Save figure
filename = 'fourier_transform_magnitude_h_g.jpg';
saveas(gcf, filename);

product = Hu .* Gu;
figure('Name', '3. Magnitude of Fourier Transform of h(m)*g(n)');
fsurf(abs(product), [-1/2 1/2]);
title('3. Magnitude of Fourier Transform of h(m)*g(n)');

% Save figure
filename = 'fourier_transform_magnitude_hm_gn.jpg';
saveas(gcf, filename);

img = im2double(imread('img\task_04\build_neoclassic.png'));
[row_size, column_size] = size(img);

% h(m)*g(n)
y1 = imfilter(imfilter(img, h, 'conv'), g.', 'conv');
% g(m)*h(n)
y2 = imfilter(imfilter(img, g, 'conv'), h.', 'conv');

figure('Name', '3.1. Y1 applied to image');
imshow(mat2gray(y1));
title('3. Y1 filter applied to image');
% Save figure
filename = 'y1.jpg';
saveas(gcf, filename);

figure('Name', '3.2. Y2 applied to image');
imshow(mat2gray(y2));
title('3. Y2 filter applied to image');
% Save figure
filename = 'y2.jpg';
saveas(gcf, filename);

Amn = y1 .^ 2 + y2 .^ 2;

% vector (y1(m,n), y2(m,n))

thetamn = abs(atan(y2 ./ y1));

% = mean of A
Amean = mean(Amn(:));
black = find(Amn <= Amean);
centroids = [0; 0.15 * pi; 0.35 * pi; 0.5 * pi];
theta = thetamn(:);

[idx, C] = kmeans(theta, 4, 'Start', centroids, 'Distance', 'cityblock');
idx(black) = 0;
cmap = [1.0 1.0 0.0;
        1.0 0.0 0.0;
        0.0 1.0 0.0;
        0.0 0.0 1.0];

L = reshape(idx, row_size, column_size);
image_label = label2rgb(L, cmap, 'k');

figure('Name', '4.Original image & Clustered(kmeans) image');
subplot(2, 1, 1), imshow(img);
title('Original image');
subplot(2, 1, 2), imshow(image_label);
title('Clustered image cmap(yellow,red,green,blue,black-background');
suptitle('4.Original image & Clustered(kmeans) image');

% Save figure
filename = 'clustered_image.jpg';
saveas(gcf, filename);
% disp(check_conditions(thetamn, Amean, Amn, image_label, C));

% Description:
% Check if the classification is correct. (Redundant)
function [check] = check_conditions(thetamn, Amean, A, img, C)
    check = true;
    red = [255; 0; 0];
    blue = [0; 0; 255];
    yellow = [255; 255; 0];
    green = [0; 255; 0];
    black = [0; 0; 0];
    [p, r, ~] = size(img);

    for i = 1:p
        for j = 1:r
            if A(i, j) > Amean && thetamn(i, j) >= 0.5 * (C(3) + C(4)) ...
                    && all(squeeze(img(i, j, :)) ~= red)
                check = false;
            end

            if A(i, j) > Amean && thetamn(i, j) > 0.5 * (C(1) + C(2)) ...
                    && thetamn(i, j) <= 0.5 * (C(2) + C(3)) && ...
                    all(squeeze(img(i, j, :)) ~= green)
                check = false;
            end

            if A(i, j) > Amean && (thetamn(i, j) > 0.5 * (C(2) + C(3)) ...
                    && thetamn(i, j) < 0.5 * (C(3) + C(4))) ...
                    && all(squeeze(img(i, j, :)) ~= blue)
                check = false;
            end

            if A(i, j) > Amean && thetamn(i, j) <= 0.5 * (C(1) + C(2)) ...
                    && all(squeeze(img(i, j, :)) ~= yellow)
                check = false;
            end

            if A(i, j) > Amean && all(squeeze(img(i, j, :)) == black)
                check = false;
            end

        end
    end
    
end
