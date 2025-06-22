;; TracePro simulation macro

(define (read-center filename)
  "Read a numeric value from a file and close it."
  (let* ((f (open-input-file filename))
         (v (read f)))
    (close-input-port f)
    v))

(define (set-prism-material)
  (property:apply-material (entity 2) "New" "index1.3" (gvector 0 0 0)))

(define (apply-source)
  (edit:add-selection (tools:face-in-body 3 (entity 1)))
  (property:apply-flux-surface-source (tools:face-in-body 3 (entity 1)) 0.3 1000 2 #f))

(define (setup-candela)
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
  (raytrace:set-radiometric-units-radiometric))

(define (simulate)
  (insert:part "prism_sat_file-sim.SAT")
  (view:profiles "xy")
  (view:zoom-all)
  (set-prism-material)
  (apply-source)
  (setup-candela)

  ;; simulation loop parameters
  (define ang_ini 10)
  (define end 80)
  (define angle 10)
  (define ini (- ang_ini angle))

  (define center_y (read-center "center_y.txt"))
  (define center_x (read-center "center_x.txt"))
  (define center_z 0)

  (define output_path "./")

  (do ((i (+ ini angle) (+ i angle))) ((> i end))
    (define angstr (number->string i))
    (entity:rotate (entity 1) center_x center_y center_z 0 0 1 angle)
    (raytrace:source)
    (let ((bmp_path (string-append output_path "view-" angstr ".bmp"))
          (txt_path (string-append output_path "polar-" angstr ".txt"))
          (rect_bmp (string-append output_path "RCD-" angstr ".bmp")))
      (file:save-as bmp_path)
      (analysis:candela-save-bmp "rectangular-distribution" rect_bmp)
      (analysis:candela-save-txt "polar-distribution" txt_path 361)))

  ;; create completion marker

  
  (file:save-as "completion_signal.OML")
  (file:save-as "../../Macro/completion_signal.OML"))

(simulate)