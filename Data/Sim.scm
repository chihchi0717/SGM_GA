(define Macro1
  (lambda ()
    (insert:part "C:/Users/user/Desktop/NTHU/MasterThesis/GA/SGM_GA/file/prism_sat_file0523-sim.SAT")
    (view:profiles "xy")
    (view:zoom-all)

    ;; 設定稜鏡材質
    (do ((i 2 (+ i 1))) ((> i 2))
      (edit:add-selection (entity i))
      (property:apply-material (entity i) "SCHOTT" "BK7" (gvector 0 0 0))
      (edit:clear-selection))

    ;; 設定光源
    (edit:add-selection (tools:face-in-body 3 (entity 1)))
    (property:apply-flux-surface-source (tools:face-in-body 3 (entity 1)) 100 1000 2 #f)

    ;; Candela Plot 設定
    (analysis:candela-normal (gvector 0 -1 0))
    (analysis:candela-up (gvector -1 0 0))
    (analysis:candela-ray-type "missed")
    (analysis:candela-symmetry "none")
    (analysis:candela-distribution 1 #f 540 #t #f)
    (analysis:candela-distribution-luminaire 180)
    (analysis:candela-distribution-max #f 0)
    (analysis:candela-distribution-min #f 0)
    (analysis:candela-distribution-log-plot #t)
    (analysis:candela-rect-distribution-angular-width 180)
    (raytrace:set-radiometric-units-photometric)

    ;; 主迴圈
    (define ang_ini 10) ; 初始角度
    (define end 80) ; 結束角度
    (define angle 10) ; 每次增加的角度
    (define ini (- ang_ini angle))
    (define center_x 0)
    (define center_y 13.2)
    (define center_z 0)
    (define output_path "C:/Users/user/Desktop/NTHU/MasterThesis/GA/SGM_GA/file/output")

    (do ((i (+ ini angle) (+ i angle))) ((> i end))
      (define angstr (number->string i)) ; <-- 修正這裡

      (entity:rotate (entity 1) center_x center_y center_z 0 0 1 angle)
      (raytrace:source)

      (let* (
        (bmp_path (string-append output_path "view-" angstr ".bmp"))
        (txt_path (string-append output_path "polar-" angstr ".txt"))
        (polar_bmp (string-append output_path "PCD-" angstr ".bmp"))
        (rect_bmp (string-append output_path "RCD-" angstr ".bmp"))
      )

        ;; 存檔命令
        (file:save-as bmp_path)
        ;; (analysis:candela-save-bmp "polar-distribution" polar_bmp)
        ;; (analysis:candela-save-bmp "rectangular-distribution" rect_bmp)
        (analysis:candela-save-txt "polar-distribution" txt_path 361)
      )
    )

    ;; 完成標記
    (file:save-as "C:/Users/user/Desktop/NTHU/MasterThesis/GA/SGM_GA/file/completion_signal.OML")
  )
)

(Macro1)
