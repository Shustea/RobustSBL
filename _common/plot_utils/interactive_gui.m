function interactive_gui(snr_list, true_DOAs, theta_grid, A_grid, A_true, K, M, nu_t_df)
% INTERACTIVE_GUI_SBL  Interactive DOA spectra viewer vs SNR with SCM/Huber/Tyler/t-Student + Bartlett/MVDR/MUSIC/SBL
%
% snr_list  : vector of SNRs in dB
% true_DOAs : vector of true angles (deg)
% theta_grid: grid (deg) matching columns of A_grid
% A_grid    : steering matrix over theta_grid, size [M x G]
% A_true    : steering matrix at true DOAs, size [M x D]
% K, M      : snapshots, sensors
% nu_t_df   : (optional) DOF for Student-t noise mixing (default 3)

if nargin < 8 || isempty(nu_t_df), nu_t_df = 3; end
D = numel(true_DOAs);
G = numel(theta_grid);
S = numel(snr_list);

% --- Covariance estimators ---
cov_names = {'SCM','Huber','Tyler','t-Student'};
num_cov = numel(cov_names);

% --- DOA methods ---
doa_names = {'Bartlett','MVDR','MUSIC','SBL'};
num_doa = numel(doa_names);

% Preallocate
Pdb = cell(num_cov,num_doa);
for ci=1:num_cov, for di=1:num_doa, Pdb{ci,di} = nan(S,G); end, end

% Fixed signals & noise
rng(0);
S_fixed = (randn(D,K)+1j*randn(D,K))/sqrt(2);
W_fixed = (randn(M,K)+1j*randn(M,K))/sqrt(2);
chi2draws = gamrnd(nu_t_df/2,2,1,K);
wt = sqrt(nu_t_df ./ chi2draws);

% Options for SBL
opts = SBLSet;
opts.Nsource = D;
opts.convergenceMu = 1;
opts.activeIndices = 1;
opts.activeIndRepN = 10;
opts.convergence.min_iteration = opts.activeIndRepN;
degRes = theta_grid(2)-theta_grid(1);
errorDOAsepP = max(1,floor(5/degRes)-1);
errorDOApeak = D+2;

% --- Loop over SNR ---
for si=1:S
    SNRi = snr_list(si);
    S_sig = S_fixed .* db2mag(SNRi);
    W = W_fixed .* wt;
    X = A_true*S_sig + W;

    % Covariance estimates
    R_scm = scm_scatter(X);
    R_h   = huber_scatter(X,0.9);
    R_tyl = tyler_scatter(X);
    R_t   = t_student_scatter(X,2.1);
    Rlist = {R_scm, R_h, R_tyl, R_t};

    for ci=1:num_cov
        R = Rlist{ci};
        % Bartlett
        Pdb{ci,1}(si,:) = 10*log10(max(bartlett_doa_estimation(A_grid,R),1e-16));
        % MVDR
        Pdb{ci,2}(si,:) = 10*log10(max(mvdr_doa_estimation(A_grid,R),1e-16));
        % MUSIC
        Pdb{ci,3}(si,:) = 10*log10(max(music_doa_estimation(A_grid,R,D),1e-16));
    end

    % --- SBL (full spectrum) ---
    [~,reportG]  = SBL_v5p12(A_grid,X,'SBL-G',inf,opts,errorDOApeak,errorDOAsepP);
    P_sbl = reportG.results.final_iteration.gamma;
    Pdb{1,4}(si,:) = 10*log10(max(P_sbl(:).',1e-16)); % treat as "SCM+SBL"

    [~,reportH]  = SBL_v5p12(A_grid, X, 'SBL-H', 0.9, opts, errorDOApeak, errorDOAsepP);
    P_sbl = reportH.results.final_iteration.gamma;
    Pdb{2,4}(si,:) = 10*log10(max(P_sbl(:).',1e-16)); % treat as "SCM+SBL"

    [~,reportTyl]  = SBL_v5p12(A_grid, X, 'SBL-Tyl', M, opts, errorDOApeak, errorDOAsepP);
    P_sbl = reportTyl.results.final_iteration.gamma;
    Pdb{3,4}(si,:) = 10*log10(max(P_sbl(:).',1e-16)); % treat as "SCM+SBL"

    [~,reportT]  = SBL_v5p12(A_grid, X, 'SBL-T', 2.1, opts, errorDOApeak, errorDOAsepP);
    P_sbl = reportT.results.final_iteration.gamma;
    Pdb{4,4}(si,:) = 10*log10(max(P_sbl(:).',1e-16)); % treat as "SCM+SBL"
end

% -------- Build UI --------
fig = figure('Name','Interactive DOA GUI with SBL','NumberTitle','off');
ax  = axes('Parent',fig); hold(ax,'on'); grid(ax,'on'); box(ax,'on');
ax.XLim = [theta_grid(1),theta_grid(end)];
xlabel(ax,'Angle (deg)'); ylabel(ax,'Power (dB)');
for dIdx=1:D, xline(ax,true_DOAs(dIdx),'r--','Ground Truth'); end

snr_idx=1;
ln=gobjects(num_cov,num_doa);
for ci=1:num_cov
    for di=1:num_doa
        ln(ci,di)=plot(ax,theta_grid,Pdb{ci,di}(snr_idx,:),...
            'LineWidth',2,'DisplayName',[doa_names{di} ' (' cov_names{ci} ')']);
    end
end
ttl=title(ax,sprintf('Spectra @ SNR = %d dB',snr_list(snr_idx)));

state.Pdb=Pdb; state.theta_grid=theta_grid; state.snr_list=snr_list;
state.lines=ln; state.title=ttl;
state.sel_cov=true(1,num_cov); state.sel_doa=true(1,num_doa);
setappdata(fig,'state',state);

% Covariance panel
panel1=uipanel('Parent',fig,'Units','normalized','Position',[0.82 0.55 0.16 0.35],...
    'Title','Covariance','FontSize',10);
for ci=1:num_cov
    uicontrol('Parent',panel1,'Style','checkbox','Units','normalized',...
        'Position',[0.05 1-ci*0.22 0.9 0.2],'String',cov_names{ci},...
        'Value',1,'Callback',@(src,~) toggleCov(ci,src));
end

% DOA panel
panel2=uipanel('Parent',fig,'Units','normalized','Position',[0.82 0.15 0.16 0.35],...
    'Title','DOA Methods','FontSize',10);
for di=1:num_doa
    uicontrol('Parent',panel2,'Style','checkbox','Units','normalized',...
        'Position',[0.05 1-di*0.22 0.9 0.2],'String',doa_names{di},...
        'Value',1,'Callback',@(src,~) toggleDOA(di,src));
end

% SNR slider
uicontrol('Style','text','Units','normalized',...
    'Position',[0.15 0.02 0.15 0.045],'String','SNR (dB):','FontSize',11);
uicontrol('Style','slider','Units','normalized','Position',[0.30 0.02 0.50 0.045],...
    'Min',1,'Max',S,'Value',snr_idx,...
    'SliderStep',[1/max(1,S-1),min(1,2/max(1,S-1))],'Callback',@onSlide);

% --- Callbacks ---
    function toggleCov(ci,src)
        st=getappdata(fig,'state'); st.sel_cov(ci)=logical(get(src,'Value'));
        setappdata(fig,'state',st); updateVis();
    end
    function toggleDOA(di,src)
        st=getappdata(fig,'state'); st.sel_doa(di)=logical(get(src,'Value'));
        setappdata(fig,'state',st); updateVis();
    end
    function updateVis()
        st=getappdata(fig,'state');
        for ci=1:num_cov, for di=1:num_doa
            if st.sel_cov(ci)&&st.sel_doa(di), set(st.lines(ci,di),'Visible','on');
            else, set(st.lines(ci,di),'Visible','off'); end
        end, end
    end
    function onSlide(src,~)
        st=getappdata(fig,'state');
        idx=max(1,min(numel(st.snr_list),round(get(src,'Value'))));
        for ci=1:num_cov, for di=1:num_doa
            set(st.lines(ci,di),'YData',st.Pdb{ci,di}(idx,:));
        end, end
        set(st.title,'String',sprintf('Spectra @ SNR = %d dB',st.snr_list(idx)));
        drawnow;
    end
end
