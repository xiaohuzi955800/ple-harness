import sys
import math
import ifcopenshell
import ifcopenshell.guid
import ifcopenshell.validate

OUT_PATH = sys.argv[1] if len(sys.argv) > 1 else "output/model.ifc"

model = ifcopenshell.file(schema='IFC4')

project = model.create_entity('IfcProject', GlobalId=ifcopenshell.guid.new(), Name='DataHallProject')
length_unit = model.create_entity('IfcSIUnit', UnitType='LENGTHUNIT', Prefix='MILLI', Name='METRE')
plane_angle_unit = model.create_entity('IfcSIUnit', UnitType='PLANEANGLEUNIT', Name='RADIAN')
unit_assignment = model.create_entity('IfcUnitAssignment', Units=[length_unit, plane_angle_unit])
project.UnitsInContext = unit_assignment

subcontexts = model.by_type('IfcGeometricRepresentationSubContext')
if subcontexts:
    context = subcontexts[0]
else:
    context_3d = model.create_entity('IfcGeometricRepresentationContext', ContextIdentifier='Model', ContextType='Model', CoordinateSpaceDimension=3, Precision=0.01, WorldCoordinateSystem=model.create_entity('IfcAxis2Placement3D', Location=model.create_entity('IfcCartesianPoint', Coordinates=[0.0, 0.0, 0.0])))
    context = model.create_entity('IfcGeometricRepresentationSubContext', ContextIdentifier='Body', ContextType='Model', ParentContext=context_3d, TargetView='MODEL_VIEW')

site = model.create_entity('IfcSite', GlobalId=ifcopenshell.guid.new(), Name='Site')
building = model.create_entity('IfcBuilding', GlobalId=ifcopenshell.guid.new(), Name='DataHallBuilding')
storey = model.create_entity('IfcBuildingStorey', GlobalId=ifcopenshell.guid.new(), Name='Ground Floor', Elevation=0.0)

model.create_entity('IfcRelAggregates', GlobalId=ifcopenshell.guid.new(), RelatingObject=project, RelatedObjects=[site])
model.create_entity('IfcRelAggregates', GlobalId=ifcopenshell.guid.new(), RelatingObject=site, RelatedObjects=[building])
model.create_entity('IfcRelAggregates', GlobalId=ifcopenshell.guid.new(), RelatingObject=building, RelatedObjects=[storey])

storey_placement = model.create_entity('IfcLocalPlacement', RelativePlacement=model.create_entity('IfcAxis2Placement3D', Location=model.create_entity('IfcCartesianPoint', Coordinates=[0.0, 0.0, 0.0])))
site.ObjectPlacement = model.create_entity('IfcLocalPlacement', RelativePlacement=model.create_entity('IfcAxis2Placement3D', Location=model.create_entity('IfcCartesianPoint', Coordinates=[0.0, 0.0, 0.0])))
building.ObjectPlacement = model.create_entity('IfcLocalPlacement', PlacementRelTo=site.ObjectPlacement, RelativePlacement=model.create_entity('IfcAxis2Placement3D', Location=model.create_entity('IfcCartesianPoint', Coordinates=[0.0, 0.0, 0.0])))
storey.ObjectPlacement = storey_placement

def create_box(w, d, h, cx, cy, z, name="Box", obj_type="Box", is_segment=False, svc=None):
    profile = model.create_entity('IfcRectangleProfileDef', ProfileType='AREA', XDim=float(w), YDim=float(d))
    direction = model.create_entity('IfcDirection', DirectionRatios=[0.0, 0.0, 1.0])
    solid = model.create_entity('IfcExtrudedAreaSolid', SweptArea=profile, ExtrudedDirection=direction, Position=model.create_entity('IfcAxis2Placement3D', Location=model.create_entity('IfcCartesianPoint', Coordinates=[0.0, 0.0, 0.0])), Depth=float(h))
    item = model.create_entity('IfcShapeRepresentation', ContextOfItems=context, RepresentationIdentifier='Body', RepresentationType='SweptSolid', Items=[solid])
    shape = model.create_entity('IfcProductDefinitionShape', Representations=[item])
    placement = model.create_entity('IfcLocalPlacement', PlacementRelTo=storey_placement, RelativePlacement=model.create_entity('IfcAxis2Placement3D', Location=model.create_entity('IfcCartesianPoint', Coordinates=[float(cx), float(cy), float(z)])))
    
    if is_segment:
        if svc in ('tray', 'busway'):
            elem = model.create_entity('IfcCableCarrierSegment', GlobalId=ifcopenshell.guid.new(), Name=name, ObjectType=obj_type, ObjectPlacement=placement, Representation=shape)
        else:
            elem = model.create_entity('IfcPipeSegment', GlobalId=ifcopenshell.guid.new(), Name=name, ObjectType=obj_type, ObjectPlacement=placement, Representation=shape)
    else:
        elem = model.create_entity('IfcBuildingElementProxy', GlobalId=ifcopenshell.guid.new(), Name=name, ObjectType=obj_type, ObjectPlacement=placement, Representation=shape)
    return elem

# 1. Floor Slab
create_box(27000, 21000, 300, 13500, 10500, -300, name="Slab", obj_type="Slab")

# 2. Columns
columns_data = [
    (8442, 8370), (16808, 8370), (25213, 8370),
    (8442, 16758), (16808, 16758), (25213, 16758)
]
for i, (cx, cy) in enumerate(columns_data):
    create_box(600, 600, 3600, cx, cy, 0, name=f"Column-{i+1}", obj_type="Column")

# 3. CRAHs
crahs_data = [
    (9622, 467, "CRAH-02"), (17402, 467, "CRAH-03"), (1236, 469, "CRAH-01"), (25783, 469, "CRAH-04"),
    (4528, 2996, "CRAH-A01"), (15914, 2997, "CRAH-A03"), (9515, 3001, "CRAH-A02"),
    (4527, 5992, "CRAH-B01"), (9712, 5996, "CRAH-B02"), (16528, 5998, "CRAH-B03"),
    (3929, 8378, "CRAH-C01"), (9327 + 15, 8380, "CRAH-C02"), (14923, 8396, "CRAH-C03"),
    (4527, 11011, "CRAH-D01"), (16525, 11374, "CRAH-D03"), (9713, 11376, "CRAH-D02"),
    (4521, 13734, "CRAH-E01"), (9509, 13740, "CRAH-E02"), (15922, 13740, "CRAH-E03"),
    (4528, 16765, "CRAH-F01"), (17516 + 192, 16771, "CRAH-F03"), (9329 + 13, 16775, "CRAH-F02")
]
for cx, cy, tag in crahs_data:
    create_box(1200, 900, 2500, cx, cy, 0, name=tag, obj_type="CRAH")

# 4. PDUs
pdus_data = [
    (21410, 2797, "PDU-01"), (21984, 6160, "PDU-02"), (21597, 8174, "PDU-03"),
    (21979, 11563, "PDU-04"), (21405, 13556, "PDU-05"), (22397, 16950, "PDU-06")
]
for cx, cy, tag in pdus_data:
    create_box(800, 1000, 2000, cx, cy, 0, name=tag, obj_type="PDU")

# 5. Cabinets
rows_data = [
    ("A", 3000, 1800, 19200),
    ("B", 6000, 1800, 19800),
    ("C", 9200, 1800, 19200),
    ("D", 11400, 1787, 19800),
    ("E", 13800, 1800, 19200),
    ("F", 17600, 1800, 19200)
]

orig_y_map = {3000: 3000, 6000: 6000, 9200: 8400, 11400: 11400, 13800: 13800, 17600: 16800}

for r_name, y_c, x_start, x_end in rows_data:
    orig_y = orig_y_map[y_c]
    x = x_start + 300
    cabinet_idx = 1
    while x < x_end - 300:
        skip = False
        for ccx, ccy, _tag in crahs_data:
            if abs(x - ccx) < 900 and abs(orig_y - ccy) < 950:
                skip = True
                break
        for col_x, col_y in columns_data:
            if abs(x - col_x) < 600 and abs(orig_y - col_y) < 600:
                skip = True
                break
        if not skip:
            create_box(600, 1000, 2000, x, orig_y, 0, name=f"Rack-{r_name}-{cabinet_idx:02d}", obj_type="Rack")
            cabinet_idx += 1
        x += 600

# 6. MEP Routes
create_box(683, 19000, 127, 500, 9750, 2940, name="Tray-WestTrunk", obj_type="CableTray", is_segment=True, svc="tray")
create_box(128, 19000, 147, 1100, 9750, -600, name="Busway-WestTrunk", obj_type="Busway", is_segment=True, svc="busway")
create_box(168.3, 19000, 168.3, 1500, 9750, 2600, name="CHW-WestTrunk", obj_type="CHW", is_segment=True, svc="chw")

south_perim_len = 26000 - 1584.15
create_box(south_perim_len, 168.3, 168.3, (1584.15 + 26000) / 2, 1000, 2600, name="CHW-SouthPerimeter", obj_type="CHW", is_segment=True, svc="chw")

orig_to_header_y = {3000: 3000, 6000: 6000, 8400: 9200, 11400: 11400, 13800: 13800, 16800: 17600}

TRAY_BUSWAY_ENDS = {'A': 14679, 'B': 16450, 'C': 14679, 'D': 16400, 'E': 14679, 'F': 16750}
CHW_ENDS = {'A': 15914, 'B': 16528, 'C': 14923, 'D': 16525, 'E': 15922, 'F': 17516}

for r_name, y_c, x_start, x_end in rows_data:
    full_len = x_end - x_start
    opt_len = TRAY_BUSWAY_ENDS[r_name] - x_start
    x_opt_end = x_start + opt_len
    cx_mid_opt = (x_start + x_opt_end) / 2
    cx_mid_full = (x_start + x_end) / 2
    orig_y = orig_y_map[y_c]
    
    tray_y = y_c + 600 if r_name in ('C', 'F') else y_c - 500
    create_box(opt_len, 683, 127, cx_mid_opt, tray_y, 2940, name=f"Tray-Row-{r_name}", obj_type="CableTray", is_segment=True, svc="tray")
    feed_start_x = 500 + 341.5
    feed_len_t = x_start - feed_start_x
    create_box(feed_len_t, 683, 127, (feed_start_x + x_start) / 2, tray_y, 2940, name=f"Tray-Feed-{r_name}", obj_type="CableTray", is_segment=True, svc="tray")

    create_box(opt_len, 128, 147, cx_mid_opt, y_c, -600, name=f"Busway-Row-{r_name}", obj_type="Busway", is_segment=True, svc="busway")
    b_start_x = 1100 + 64
    b_len = x_start - b_start_x
    create_box(b_len, 128, 147, (b_start_x + x_start) / 2, y_c, -600, name=f"Busway-Feed-{r_name}", obj_type="Busway", is_segment=True, svc="busway")

    # CHW row header full length for connectivity
    chw_len = CHW_ENDS[r_name] - x_start
    cx_mid_chw = (x_start + CHW_ENDS[r_name]) / 2
    create_box(chw_len, 168.3, 168.3, cx_mid_chw, y_c, 2600, name=f"CHW-Row-{r_name}", obj_type="CHW", is_segment=True, svc="chw")
    c_start_x = 1500 + 84.15
    c_len = x_start - c_start_x
    create_box(c_len, 168.3, 168.3, (c_start_x + x_start) / 2, y_c, 2600, name=f"CHW-Feed-{r_name}", obj_type="CHW", is_segment=True, svc="chw")

for ccx, ccy, ctag in crahs_data:
    if ccy < 1500:
        target_y = 1000 - 84.15
        branch_len = abs(ccy - target_y)
        create_box(168.3, branch_len, 168.3, ccx, (target_y + ccy) / 2, 2600, name=f"CHW-Branch-{ctag}", obj_type="CHW", is_segment=True, svc="chw")
    else:
        orig_y = next(r[1] for r in [("A", 3000), ("B", 6000), ("C", 8400), ("D", 11400), ("E", 13800), ("F", 16800)] if abs(ccy - r[1]) < 800)
        hdr_y = orig_to_header_y[orig_y]
        target_y = hdr_y - 84.15 if ccy < hdr_y else hdr_y + 84.15
        if abs(ccy - hdr_y) > 84.15:
            branch_len = abs(ccy - target_y)
            create_box(168.3, branch_len, 168.3, ccx, (target_y + ccy) / 2, 2600, name=f"CHW-Branch-{ctag}", obj_type="CHW", is_segment=True, svc="chw")
    create_box(50.0, 50.0, 100.0, ccx, ccy, 2500.0, name=f"CHW-Drop-{ctag}", obj_type="CHW", is_segment=True, svc="chw")

model.write(OUT_PATH)
print("Model written to", OUT_PATH)
