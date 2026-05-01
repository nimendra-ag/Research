from rdkit import Chem
from neo4j import GraphDatabase

uri = "bolt://localhost:7687"
user = "neo4j"
password = "12345678"

driver = GraphDatabase.driver(uri, auth=(user, password))

supplier = Chem.SDMolSupplier("datasets/NCI_full/1total-connect.sdf")

def clear_db(tx):
    tx.run("MATCH (n) DETACH DELETE n")

def create_molecule(tx, mol_id, label):
    tx.run("""
        CREATE (m:Molecule {id:$id, label:$label})
    """, id=mol_id, label=label)

def create_atom(tx, mol_id, atom_idx, symbol):
    tx.run("""
        MATCH (m:Molecule {id:$mid})
        CREATE (a:Atom {atom_id:$aid, symbol:$symbol, molecule_id:$mid})
        CREATE (m)-[:HAS_ATOM]->(a)
    """, mid=mol_id, aid=atom_idx, symbol=symbol)

def create_bond(tx, mol_id, a1, a2, bond_type):
    tx.run("""
        MATCH (x:Atom {atom_id:$a1, molecule_id:$mid})
        MATCH (y:Atom {atom_id:$a2, molecule_id:$mid})
        CREATE (x)-[:BOND {type:$btype}]-(y)
    """, a1=a1, a2=a2, mid=mol_id, btype=bond_type)

def main():
    with driver.session() as session:
        session.execute_write(clear_db)

        count = 0

        for idx, mol in enumerate(supplier):
            if mol is None:
                continue

            mol_id = idx
            label = mol.GetProp("value") if mol.HasProp("value") else "unknown"

            session.execute_write(create_molecule, mol_id, label)

            # atoms
            for atom in mol.GetAtoms():
                session.execute_write(
                    create_atom,
                    mol_id,
                    atom.GetIdx(),
                    atom.GetSymbol()
                )

            # bonds
            for bond in mol.GetBonds():
                session.execute_write(
                    create_bond,
                    mol_id,
                    bond.GetBeginAtomIdx(),
                    bond.GetEndAtomIdx(),
                    int(bond.GetBondTypeAsDouble())
                )

            count += 1

            if count == 100:   # start with first 100 molecules
                break

    print("Imported", count, "molecules")
    driver.close()

if __name__ == "__main__":
    main()