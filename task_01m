% Myrsini Gkolemi

% let 1 <= m <= M, 1 <= n <= N
% let a < 1

% ImageDatastore object to manage a collection of image files

% close all;

imgs = imageDatastore('img\task_01');
imgs_length = length(imgs.Files);
image_data = cell(1, imgs_length);

for i = 1:imgs_length
    img = im2double(readimage(imgs, i));
    image_data{i} = img;
end

% factor
a = 1 / 8;

% Matrix with (G,E) pairs for each image(helping structure)
%
% index(=img_1)     G E   G E   GE
% .                 G E   G E   GE
% .                  .
% .                  .
% .                  .
% index(=img_n-1)   G E   G E   GE
% column[1 2] = NN
% column[3 4] = NN_antialiasing
% column[5 6] = NN_antialiasing_bicubic

% initialise with zeros
matrix_g_e = zeros(imgs_length, 6);

for i = 1:imgs_length
    img = image_data{i};
    img_1 = NN(img, a);
    img_2 = NN_antialiasing(img, a);
    img_3 = NN_antialiasing_bicubic(img, a);

    processed_images = {img_1, img_2, img_3};
    show_images(i, img, img_1, img_2, img_3);

    for j = 1:3
        g_e = [get_G(get_Gv(img, processed_images{j}), get_Gh(img, ...
                   processed_images{j})) get_E(img, processed_images{j})];
        matrix_g_e(i, (j * 2 - 1):j * 2) = g_e;
    end

end

% ---------------------Graphic representation of G,E ----------------------
% Use mat_g_e
% E = a1*G + a0
% For each subquery(1 to 3) plot points from sample folder in G,E graph.
% Plot
% (polyfit polyval)

for i = 1:3
    plot_GE(i, matrix_g_e);
end

% ------------------------------Functions----------------------------------

% 1.
% nearest neighbor - nearest neighbor
% If method is 'nearest', then the default value of 'Antialiasing' is false

function [f_img] = NN(img, a)
    r_img = imresize(img, a, 'nearest');
    f_img = imresize(r_img, 1 / a, 'nearest');
end

% 2.
% (nearest neighbor - antialiasing) - nearest neighbor
function [f_img] = NN_antialiasing(img, a)
    r_img = imresize(img, a, 'nearest', 'Antialiasing', true);
    f_img = imresize(r_img, 1 / a, 'nearest');
end

% 3.
% Note: Bicubic interpolation can produce pixel values outside the original
% range.
% (nearest neighbor - antialiasing) - bicubic
function [f_img] = NN_antialiasing_bicubic(img, a)
    r_img = imresize(img, a, 'nearest', 'Antialiasing', true);
    f_img = imresize(r_img, 1 / a, 'bicubic');
end

function [S] = get_S(img)
    %   alternative solution
    %   S = 0;
    %   [M,N,~] = size(img);
    %
    %   for m = 1:M
    %     for n = 1:N
    %         S = S + img(m,n)^2;
    %     end
    %   end
    %   Can't use sum(A, 'all') 2018b
    S = sum(sum(img .^ 2));
end

function [E] = get_E(img_x, img_y)
    %   alternative solution
    E = 0;
    [M, N, ~] = size(img_x);
    for m = 1:M
        for n = 1:N
            E = E + ((img_x(m, n) - img_y(m, n)) ^ 2);
        end
    end
    E = (1 / get_S(img_x)) * E;
    %   E = (1/get_S(img_x)) * sum(sum((img_x - img_y).^2));
end

function [Gh] = get_Gh(img_x, img_y)
    Gh = 0;
    [M, N, ~] = size(img_x);
    for m = 1:M
        for n = 2:N
            Gh = Gh + ((img_x(m, n) - img_y(m, n - 1)) ^ 2);
        end
    end
    Gh = (1 / get_S(img_x)) * Gh;
end

function [Gv] = get_Gv(img_x, img_y)
    Gv = 0;
    [M, N, ~] = size(img_x);
    for m = 2:M
        for n = 1:N
            Gv = Gv + ((img_x(m, n) - img_y(m - 1, n)) ^ 2);
        end
    end
    Gv = (1 / get_S(img_x)) * Gv;
end

function [G] = get_G(Gv, Gh)
    G = max(Gv, Gh);
end

function [] = plot_GE(i, matrix_g_e)
    figure('Name', [num2str(i), '. (G,E) / Approximation']);
    G = matrix_g_e(:, i * 2 - 1);
    E = matrix_g_e(:, i * 2);
    plot(G, E, 'o');
    hold on;
    grid on;
    p = polyfit(G, E, 1); %degree one, line
    f = polyval(p, G);
    plot(f, G, '-');
    legend('(G,E)', 'linear regression', 'Location', 'northwest');
    title([num2str(i), '.']);
    xlabel('G');
    ylabel('E');
    hold off;
    filename = [num2str(i), '_GE_approximation.jpg'];
    saveas(gcf, filename);
end

function [] = show_images(i, img, img_1, img_2, img_3)
    figure('Name', ['Image ', num2str(i), '.']);
    subplot(2, 2, 1), imshow(img);
    title('1.Original image');
    subplot(2, 2, 2), imshow(img_1);
    title('2.NN');
    subplot(2, 2, 3), imshow(img_2);
    title('3.NN-Antial');
    subplot(2, 2, 4), imshow(img_3);
    title('4.NN-Antial-Bicub');
    filename = ['image_', num2str(i), '.jpg'];
    saveas(gcf, filename);
end
