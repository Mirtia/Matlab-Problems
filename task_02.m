% Myrsini Gkolemi

p_e = load('mat\p_e.mat');
p_G = load('mat\p_G.mat');
p_R = load('mat\p_R.mat');

% Parse image files

imgs = imageDatastore('img\task_02');
imgs_length = length(imgs.Files);
image_data = cell(1, imgs_length);

% Read images from folder
for i = 1:imgs_length
    img = readimage(imgs, i);
    image_data{i} = img;
end

for i = 1:imgs_length

    T1 = get_T1_all(image_data{i}, p_G, p_e);
    T2 = get_T2_all(image_data{i}, p_R, p_e);
    v_T1 = apply_Tn(image_data{i}, T2);
    v_T2 = apply_Tn(image_data{i}, T1);
    [J, v_T] = apply_T(image_data{i});
    display_T(v_T, T1, T2, i);
    hist_img = get_adapteq_image(image_data{i});
    thresh_img = get_otsu_image(image_data{i});

    plot_transforms(image_data{i}, i, J, v_T1, v_T2);
    plot_histogram(img, hist_img, i);
    plot_otsu(thresh_img, i);
end

close all
% ------------------------------Functions----------------------------------

function [n_pdf] = normal_pdf(img)
    % imhist(img);
    n_pdf = imhist((img)) ./ numel(img);
end

% PDF
function [p_l] = probability_density_function(l, img)
    % l must be in the range of [0 255] -> [1 256](indexing)
    if (l >= 1 && l <= 256)
        normalized_pdf = normal_pdf(img);
        p_l = normalized_pdf(l);
    else
        disp("l must be in the range of [1 256]");
        p_l = -1;
    end

end

% CDF
function [q_l] = probability_distribution_function(l, img)

    if (l >= 1 && l <= 256)
        normalized_pdf = normal_pdf(img);
        s_c = cumsum(normalized_pdf(1:l));
        q_l = s_c(l);
    else
        disp("l must be in the range of [1 256]");
        q_l = -1;
    end

end

% pi_0
function [pi_0] = get_pi_0(img, p_e)
    pi_0 = probability_density_function(1, img) / p_e.p_e(1);
end

% p1(l)
function [p_1] = get_p1(l, img, p_G, p_e)
    pi_0 = get_pi_0(img, p_e);
    p_1 = pi_0 * p_e.p_e(l) + (1 - pi_0) * p_G.p_G(l);
end

% p2(l)
function [p_2] = get_p2(l, img, p_R, p_e)
    pi_0 = get_pi_0(img, p_e);
    p_2 = pi_0 * p_e.p_e(l) + (1 - pi_0) * p_R.p_R(l);
end

% gaussian PDF
function [p_g] = gaussian_density_function(l, img, p_G, p_e)

    if (l >= 1 && l <= 256)
        p_g = get_p1(l, img, p_G, p_e);
    else
        disp("l must be in the range of [1 256]");
        p_g = -1;
    end

end

% rice PDF
function [p_r] = rice_density_function(l, img, p_R, p_e)

    if (l >= 1 && l <= 256)
        p_r = get_p2(l, img, p_R, p_e);
    else
        disp("l must be in the range of [1 256]");
        p_r = -1;
    end

end

% gaussian CDF
function [q_g] = gaussian_distribution_function(l, img, p_G, p_e)

    if (l >= 1 && l <= 256)
        q_g = 0;

        for i = 1:l
            q_g = q_g + get_p1(i, img, p_G, p_e);
        end

    else
        disp("l must be in the range of [1 256]");
        q_g = -1;
    end

end

% rice CDF
function [q_r] = rice_distribution_function(l, img, p_R, p_e)

    if (l >= 1 && l <= 256)
        q_r = 0;

        for i = 1:l
            q_r = q_r + get_p2(i, img, p_R, p_e);
        end

    else
        disp("l must be in the range of [1 256]");
        q_r = -1;
    end

end

% T1 transformation
function [T1_l] = get_T1(l, img, p_G, p_e)

    if (l >= 1 && l <= 256)
        Q = zeros(256, 1);
        Q(1) = get_p1(1, img, p_G, p_e);

        for i = 2:256
            Q(i) = Q(i - 1) + get_p1(i, img, p_G, p_e);
        end

        pdf = probability_distribution_function(l, img);
        difference = abs(Q - pdf);
        minimum = min(difference);
        T1_l = find (difference == minimum);
    else
        disp("l must be in the range of [1 256]");
        T1_l = -1;
    end

end

% T2 transformation
function [T2_l] = get_T2(l, img, p_R, p_e)

    if (l >= 1 && l <= 256)
        Q = zeros(256, 1);
        Q(1) = get_p2(1, img, p_R, p_e);

        for i = 2:256
            Q(i) = Q(i - 1) + get_p2(i, img, p_R, p_e);
        end

        pdf = probability_distribution_function(l, img);
        difference = abs(Q - pdf);
        minimum = min(difference);
        T2_l = find (difference == minimum);
    else
        disp("l must be in the range of [1 256]");
        T2_l = -1;
    end

end

% T1 transformation for each l
function [T1] = get_T1_all(img, p_G, p_e)
    T1 = zeros(256, 1);

    for i = 1:256
        T1(i) = get_T1(i, img, p_G, p_e);
    end

end

% T2 transformation for each l
function [T2] = get_T2_all(img, p_R, p_e)
    T2 = zeros(256, 1);

    for i = 1:256
        T2(i) = get_T2(i, img, p_R, p_e);
    end

end

% Tn transformation applied to image
function [new_img] = apply_Tn(img, T)

    [M, N, ~] = size(img);
    new_img = img;

    for i = 1:M

        for j = 1:N
            ind = img(i, j);
            new_img(i, j) = T(ind + 1);
        end

    end

    %imshow(new_img);
end

% T transformation applied to image
function [J, T] = apply_T(img)
    [J, T] = histeq(img);
    T = transpose(T) * 255; % %range to [0 255]
end

% Display L mappings of Intensity Transformations
function [] = display_T(T, T1, T2, i)
    x = 0:255;
    figure('Name', 'Intensity Transformations applied to image X');
    plot(x, T);
    hold on;
    plot(x, T1);
    hold on;
    plot(x, T2);
    hold off;

    title(['Intensity Transformations [ T T1 T2 ] applied to image (', ...
               num2str(i), ')']);
    xlabel('Input Values');
    ylabel('Output Values');
    ylim([0 255]);
    xlim([0 255]);
    legend('T', 'T1', 'T2', 'Location', 'northwest');
    xticks([0 64 128 192 255]);
    yticks([0 64 128 192 255]);
    grid on;
    filename = [num2str(i), '_relation.jpg'];
    saveas(gcf, filename);
end

% Apply adapthiseq to image
function [new_img] = get_adapteq_image(img)
    new_img = adapthisteq(img, 'NumTiles', [2 2], 'clipLimit', 0.02);
end

% Apply automatic image thresholding with Otsu method
function [new_img] = get_otsu_image(img)
    new_img = imbinarize(img);
end

% Functions for plotting

function [] = plot_transforms(img, i, J, v_T1, v_T2)
    % Display:
    % 1. Original image
    % 2. Image with T1 transfrom
    % 3. Image with T2 transform
    % 4. Image with T transform
    figure('Name', ['Image ', num2str(i), '.']);
    subplot(2, 2, 1), imshow(img);
    title('Original image');
    subplot(2, 2, 2), imshow(J);
    title('T transformed image');
    subplot(2, 2, 3), imshow(v_T1);
    title('T1 transformed image');
    subplot(2, 2, 4), imshow(v_T2);
    title('T2 transformed image');
    suptitle('Transformations');
    % Save as .jpeg
    filename = [num2str(i), '_transform.jpg'];
    saveas(gcf, filename);
end

function [] = plot_histogram(img, hist_img, i)
    figure('Name', 'Image histograms');
    subplot(1, 2, 1), imhist(img);
    title('Original image');
    subplot(1, 2, 2), imhist(hist_img);
    title('Equalized image');
    suptitle('Image histograms');
    % Save as .jpeg
    filename = [num2str(i), '_histogram.jpg'];
    saveas(gcf, filename);
end

function [] = plot_otsu(img, i)
    figure('Name', ['Otsu threshold image', num2str(i)]);
    title('Otsu threshold image');
    imshow(img);
    hold off;
    % Save as .jpeg
    filename = [num2str(i), '_otsu.jpg'];
    saveas(gcf, filename);
end
