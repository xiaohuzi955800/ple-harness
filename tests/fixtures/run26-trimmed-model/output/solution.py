import sys
import os
import math
import ifcopenshell
import ifcopenshell.validate

def create_box_extrusion(model, body_context, width, depth, height, cx, cy, z):
    profile = model.create_entity(
        'IfcRectangleProfileDef',
        ProfileType='AREA',
        XDim=float(width),
        YDim=float(depth)
    )
    dir_z = model.create_entity('IfcDirection', DirectionRatios=[0.0, 0.0, 1.0])
    pos = model.create_entity(
        'IfcAxis2Placement3D',
        Location=model.create_entity('IfcCartesianPoint', Coordinates=[float(cx), float(cy), float(z)]),
        Axis=dir_z,
        RefDirection=model.create_entity('IfcDirection', DirectionRatios=[1.0, 0.0, 0.0])
    )
    solid = model.create_entity(
        'IfcExtrudedAreaSolid',
        SweptArea=profile,
        Position=pos,
        ExtrudedDirection=dir_z,
        Depth=float(height)
    )
    rep = model.create_entity(
        'IfcShapeRepresentation',
        ContextOfItems=body_context,
        RepresentationIdentifier='Body',
        RepresentationType='SweptSolid',
        Items=[solid]
    )
    return model.create_entity('IfcProductDefinitionShape', Representations=[rep])

def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "output/model.ifc"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    model = ifcopenshell.file(schema='IFC4')
    
    project = model.create_entity('IfcProject', GlobalId=ifcopenshell.guid.new(), Name='DataHallProject')
    length_unit = model.create_entity('IfcSIUnit', UnitType='LENGTHUNIT', Prefix='MILLI', Name='METRE')
    plane_angle_unit = model.create_entity('IfcSIUnit', UnitType='PLANEANGLEUNIT', Name='RADIAN')
    unit_assignment = model.create_entity('IfcUnitAssignment', Units=[length_unit, plane_angle_unit])
    project.UnitsInContext = unit_assignment

    site = model.create_entity('IfcSite', GlobalId=ifcopenshell.guid.new(), Name='Site')
    building = model.create_entity('IfcBuilding', GlobalId=ifcopenshell.guid.new(), Name='DataHallBuilding')
    storey = model.create_entity('IfcBuildingStorey', GlobalId=ifcopenshell.guid.new(), Name='GroundFloor', Elevation=0.0)

    model.create_entity('IfcRelAggregates', GlobalId=ifcopenshell.guid.new(), RelatingObject=project, RelatedObjects=[site])
    model.create_entity('IfcRelAggregates', GlobalId=ifcopenshell.guid.new(), RelatingObject=site, RelatedObjects=[building])
    model.create_entity('IfcRelAggregates', GlobalId=ifcopenshell.guid.new(), RelatingObject=building, RelatedObjects=[storey])

    geom_context = model.create_entity(
        'IfcGeometricRepresentationContext',
        ContextIdentifier='Model',
        ContextType='Model',
        CoordinateSpaceDimension=3,
        Precision=1.0e-5,
        WorldCoordinateSystem=model.create_entity('IfcAxis2Placement3D', Location=model.create_entity('IfcCartesianPoint', Coordinates=[0.0, 0.0, 0.0]))
    )
    body_context = model.create_entity(
        'IfcGeometricRepresentationSubContext',
        ContextIdentifier='Body',
        ContextType='Model',
        ParentContext=geom_context,
        TargetView='MODEL_VIEW'
    )

    def add_element(name, cls, shape, placement_coords=[0,0,0], object_type=None):
        pl = model.create_entity(
            'IfcLocalPlacement',
            RelativePlacement=model.create_entity(
                'IfcAxis2Placement3D',
                Location=model.create_entity('IfcCartesianPoint', Coordinates=[float(c) for c in placement_coords])
            )
        )
        kwargs = {
            'GlobalId': ifcopenshell.guid.new(),
            'Name': name,
            'ObjectPlacement': pl,
            'Representation': shape
        }
        if object_type:
            kwargs['ObjectType'] = object_type
        elem = model.create_entity(cls, **kwargs)
        model.create_entity(
            'IfcRelContainedInSpatialStructure',
            GlobalId=ifcopenshell.guid.new(),
            RelatingStructure=storey,
            RelatedElements=[elem]
        )
        return elem

    # 1. Slab
    slab_shape = create_box_extrusion(model, body_context, 27000, 21000, 300, 13500, 10500, -300)
    add_element('Slab', 'IfcSlab', slab_shape, object_type='FLOOR')

    # 2. Columns
    columns_data = [
        (8442, 8370), (16808, 8370), (25213, 8370),
        (8442, 16758), (16808, 16758), (25213, 16758)
    ]
    for i, (cx, cy) in enumerate(columns_data):
        c_shape = create_box_extrusion(model, body_context, 600, 600, 3600, cx, cy, 0)
        add_element(f'Column-{i+1}', 'IfcColumn', c_shape, object_type='Column')

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
    crah_objects = []
    for cx, cy, tag in crahs_data:
        c_shape = create_box_extrusion(model, body_context, 1200, 900, 2000, cx, cy, 0)
        elem = add_element(tag, 'IfcFlowTerminal', c_shape, object_type='CRAH')
        crah_objects.append((cx, cy, tag, elem))

    # 4. Racks
    rows_data = [
        ('A', 3000, 1800, 19200),
        ('B', 6000, 1800, 19800),
        ('C', 8400, 1800, 19200),
        ('D', 11400, 1800, 19800),
        ('E', 13800, 1800, 19200),
        ('F', 16800, 1800, 19800),
    ]
    rack_count = 0
    for r_name, y_c, x_s, x_e in rows_data:
        x = x_s + 300
        while x + 300 <= x_e:
            skip = False
            for ccx, ccy, _ in crahs_data:
                if abs(x - ccx) < 900 and abs(y_c - ccy) < 950:
                    skip = True
                    break
            for colx, coly in columns_data:
                if abs(x - colx) < 600 and abs(y_c - coly) < 650:
                    skip = True
                    break
            if not skip:
                r_shape = create_box_extrusion(model, body_context, 600, 1000, 2000, x, y_c, 0)
                add_element(f'Rack-{r_name}-{rack_count}', 'IfcFurniture', r_shape, object_type='Rack')
                rack_count += 1
            x += 600

    # 5. PDUs
    pdus_data = [
        (21410, 2797, "PDU-01"), (21984, 6160, "PDU-02"), (21597, 8174, "PDU-03"),
        (21979, 11563, "PDU-04"), (21405, 13556, "PDU-05"), (22397, 16950, "PDU-06")
    ]
    for px, py, ptag in pdus_data:
        p_shape = create_box_extrusion(model, body_context, 800, 800, 2000, px, py, 0)
        add_element(ptag, 'IfcElectricDistributionBoard', p_shape, object_type='PDU')

    # 6. Sprinklers
    sprinklers_data = [
        (10521, 1506), (13494, 1508), (22506, 1508), (25485, 1508), (19499, 1509),
        (7508, 1512), (16495, 1512), (1511, 1514), (4503, 1516), (7499, 4509),
        (4503, 4510), (10511, 4511), (13501, 4511), (22502, 4514), (25489, 4514),
        (16500, 4516), (19495, 4517), (1511, 4519), (4542, 7484), (16537, 7485),
        (19461, 7485), (7468, 7491), (13495, 7513), (25489, 7514), (1496, 7515),
        (22503, 7515), (10516, 7523), (16488, 10476), (4489, 10480), (22504, 10483),
        (10516, 10485), (25482, 10485), (13494, 10486), (19507, 10486), (1502, 10487),
        (7509, 10489), (1507, 13464), (13496, 13466), (22506, 13470), (25485, 13470),
        (16500, 13474), (4499, 13475), (10516, 13476), (19500, 13479), (7501, 13481),
        (25499, 16380), (4498, 16466), (19481, 16466), (22505, 16466), (10511, 16468),
        (13490, 16468), (7497, 16470), (1504, 16474), (16427, 16474), (4497, 19463),
        (16487, 19465), (25491, 19467), (10513, 19472), (19500, 19473), (1499, 19475),
        (7501, 19476), (13491, 19476), (22506, 19477)
    ]
    for i, (sx, sy) in enumerate(sprinklers_data):
        s_shape = create_box_extrusion(model, body_context, 100, 100, 50, sx, sy, 3575)
        add_element(f'Sprinkler-{i+1}', 'IfcFlowController', s_shape, object_type='Sprinkler')

    # 7. MEP Routing
    def add_segment(name, cls, w, d, h, cx, cy, z, obj_type):
        shape = create_box_extrusion(model, body_context, w, d, h, cx, cy, z)
        return add_element(name, cls, shape, object_type=obj_type)

    add_segment('Tray-WestTrunk', 'IfcCableCarrierSegment', 683, 15750, 127, 400, 8375, 2650, 'CableTray')  # B: north end 20000 -> 16250
    add_segment('Busway-WestTrunk', 'IfcCableCarrierSegment', 128, 16450, 147, 1300, 8725, -600, 'Busway')  # B: north end 20000 -> 16950
    
    chw_trunk_len = 16950 - 584.15  # run 26: north end 17350 -> 16950
    chw_trunk_cy = (584.15 + 16950) / 2.0  # run 26: north end 17350 -> 16950
    add_segment('CHW-WestTrunk', 'IfcPipeSegment', 168.3, chw_trunk_len, 168.3, 1900, chw_trunk_cy, 2946.7, 'CHW')

    # Row Headers & Feeders & Drops -- run 26 trimmed variant:
    # east ends per the CORRECTED gate (cabinet lengths + 2500 mm reach):
    # A/C/E tray+busway cut (they set no span), B/F lifted to 82.1% spans
    tray_ends = {'A': 14679, 'B': 16603, 'C': 15061, 'D': 16555, 'E': 14324, 'F': 16931}
    busway_ends = {'A': 14679, 'B': 16603, 'C': 15061, 'D': 16555, 'E': 14324, 'F': 16931}
    chw_ends = {'A': 15694, 'B': 16716, 'C': 16040, 'D': 16416, 'E': 15702, 'F': 17500}
    for r_name, y_c, x_s, x_e in rows_data:
        t_e = tray_ends[r_name]
        t_len = t_e - 1800
        t_cx = (1800 + t_e) / 2.0
        add_segment(f'Tray-Row-{r_name}', 'IfcCableCarrierSegment', t_len, 683, 127, t_cx, y_c - 1000, 2650, 'CableTray')
        feeder_len = 1800 - (400 + 341.5)
        feeder_cx = ((400 + 341.5) + 1800) / 2.0
        add_segment(f'Tray-Feeder-{r_name}', 'IfcCableCarrierSegment', feeder_len, 683, 127, feeder_cx, y_c - 1000, 2650, 'CableTray')
        b_e = busway_ends[r_name]
        b_len = b_e - 1800
        b_cx = (1800 + b_e) / 2.0
        add_segment(f'Busway-Row-{r_name}', 'IfcCableCarrierSegment', b_len, 128, 147, b_cx, y_c, -600, 'Busway')
        b_feeder_len = 1800 - (1300 + 64)
        b_feeder_cx = ((1300 + 64) + 1800) / 2.0
        add_segment(f'Busway-Feeder-{r_name}', 'IfcCableCarrierSegment', b_feeder_len, 128, 147, b_feeder_cx, y_c, -600, 'Busway')
        c_e = chw_ends[r_name]
        if r_name in ('C', 'F'):
            # jog chain clipped to the new east end; segments past it are dropped
            _bounds = [(1984.15, 8100, 0), (8100, 8900, 400), (8900, 16400, 0), (16400, 17300, 400), (17300, c_e, 0)]
            _names = ['1', 'Jog1', '2', 'Jog2', '3']
            for (_x0, _x1, _dy), _nm in zip(_bounds, _names):
                _a, _b = max(_x0, 1984.15), min(_x1, c_e)
                if _b - _a <= 0:
                    continue
                add_segment(f'CHW-Row-{r_name}-{_nm}', 'IfcPipeSegment', _b - _a, 168.3, 168.3, (_a + _b) / 2.0, y_c + _dy, 2946.7, 'CHW')
        else:
            chw_len = c_e - 1984.15
            chw_cx = (1984.15 + c_e) / 2.0
            add_segment(f'CHW-Row-{r_name}', 'IfcPipeSegment', chw_len, 168.3, 168.3, chw_cx, y_c, 2946.7, 'CHW')

    perim_len = 25783 - 1236
    perim_cx = (1236 + 25783) / 2.0
    add_segment('CHW-SouthPerimeter', 'IfcPipeSegment', perim_len, 168.3, 168.3, perim_cx, 500, 2946.7, 'CHW')

    for ccx, ccy, ctag in crahs_data:
        drop_height = 2946.7 - 2000.0
        add_segment(f'CHW-Drop-{ctag}', 'IfcPipeSegment', 168.3, 168.3, drop_height, ccx, ccy, 2000.0, 'CHW')
        if ctag == "CRAH-D01":
            branch_len = 11315.85 - 11011
            branch_cy = (11011 + 11315.85) / 2.0
            add_segment('CHW-Branch-D01', 'IfcPipeSegment', 168.3, branch_len, 168.3, ccx, branch_cy, 2946.7, 'CHW')

    class _L:
        def __init__(self): self.errors = []
        def error(self, *a, **k): self.errors.append((a, k))
    _log = _L()
    ifcopenshell.validate.validate(model, _log, express_rules=True)
    if _log.errors:
        print("Validation errors:", _log.errors[:5])
        sys.exit(1)

    model.write(out_path)
    print(f"Model written successfully to {out_path}")

if __name__ == '__main__':
    main()
