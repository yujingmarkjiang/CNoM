package main

import (
	"fmt"
	"os"

	"github.com/cnom/cnom/internal/experiments"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintf(os.Stderr, "usage: cnom-exp list|all|E1 [input.json] [output.json]\n")
		os.Exit(2)
	}
	cmd := os.Args[1]
	if cmd == "list" {
		for _, s := range experiments.Catalog() {
			fmt.Printf("%s\t%s\t%s\n", s.ID, s.Kind, s.Title)
		}
		return
	}
	in, out := "", ""
	if len(os.Args) > 2 {
		in = os.Args[2]
	}
	if len(os.Args) > 3 {
		out = os.Args[3]
	}
	run := func(id string) error {
		for _, s := range experiments.Catalog() {
			if s.ID == id {
				return experiments.Run(s, in, outOr(out, id))
			}
		}
		return fmt.Errorf("unknown experiment %s", id)
	}
	if cmd == "all" {
		for _, s := range experiments.Catalog() {
			if err := experiments.Run(s, in, outOr("", s.ID)); err != nil {
				fmt.Fprintln(os.Stderr, err)
				os.Exit(1)
			}
			fmt.Println(s.ID)
		}
		return
	}
	if err := run(cmd); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func outOr(out, id string) string {
	if out != "" {
		return out
	}
	return "runs/" + id + ".json"
}
