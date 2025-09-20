function doa_est = pickDOA(spectrum, theta_grid, D)
    [~, locs] = findpeaks(spectrum, 'SortStr', 'descend');
    if numel(locs) < D
        % pad if fewer peaks found
        locs = [locs; repmat(locs(end), D - numel(locs), 1)];
    end
    doa_est = sort(theta_grid(locs(1:D)));
end
