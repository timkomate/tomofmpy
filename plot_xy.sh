gmt set FONT_ANNOT_PRIMARY=14p
gmt set FONT_ANNOT_SECONDARY=10p
gmt set FONT_LABEL=18p
gmt set FONT_TITLE=24p,Helvetica,black
gmt set MAP_ANNOT_OFFSET=0.08c
gmt set MAP_FRAME_WIDTH=0.1c
gmt set MAP_TICK_LENGTH_PRIMARY=0.15c
gmt set FORMAT_GEO_MAP=DF
gmt set MAP_LABEL_OFFSET=0.2c
gmt set MAP_FRAME_TYPE=plain
gmt set MAP_TITLE_OFFSET=0.15c
gmt set MAP_FRAME_PEN=2p
gmt set MAP_TICK_LENGTH_SECONDARY=0.1c
gmt set FONT_TAG=24p,Helvetica,black
gmt set MAP_GRID_PEN=2p

R="-R0/120/0/120"
J="-JX16c"
filename="cart"

gmt xyz2grd "./test_syn/12.xyz" -I5/5 -Gvel.grd $R -rp -i0+o2.5,1+o2.5,2

gmt begin $filename pdf,png
  gmt basemap -B $R $J
  gmt grdimage $R $J vel.grd -Cviridis
  #gmt plot "./test_syn/9.xyz" -CgrayC -Sc.5c
  gmt colorbar -DJCB+w10c/0.5c+h -Bx0.05+l"Velocity" -By+l"km/s" -C$cpt
gmt end